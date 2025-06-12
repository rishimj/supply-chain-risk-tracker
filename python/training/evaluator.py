import pandas as pd
import numpy as np
import logging
import torch
from typing import Dict, Any, List

from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve, # Needed for _calculate_basic_metrics
    roc_curve # Needed for save_evaluation_plots
)
from sklearn.calibration import calibration_curve # Moved calibration_curve import here

import matplotlib.pyplot as plt

# It's good practice for each module to have its own logger
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """
    Handles the evaluation of a trained model against a dataset.
    """
    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the ModelEvaluator.

        Args:
            config (Dict[str, Any]): The evaluation section of the main config.
        """
        self.config = config
        # Initialize threshold here to ensure it's always set
        self.threshold = self.config.get('prediction_threshold', 0.5)

    def evaluate_model(
        self,
        model: Any,
        X_lstm: torch.Tensor,
        X_tabular: np.ndarray,
        y: pd.Series,
        metadata: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Evaluates the model's performance on the provided data.

        Args:
            model (Any): The trained model object with a .predict() method.
            X_lstm (torch.Tensor): The pre-processed time-series features for the LSTM.
            X_tabular (np.ndarray): The pre-processed tabular features.
            y (pd.Series): The true labels.
            metadata (Dict[str, Any]): Additional metadata about the data.

        Returns:
            Dict[str, Any]: A dictionary containing performance metrics and other results.
        """
        logger.info("Starting model evaluation...")
        try:
            # The model's predict method requires data in this specific two-part format.
            predictions_proba = model.predict(X_lstm=X_lstm, X_tabular=X_tabular)

            # Convert probabilities to binary predictions for classification metrics
            predictions_binary = (predictions_proba > self.threshold).astype(int)

            logger.info("Calculating performance metrics...")
            # Using the comprehensive _calculate_basic_metrics
            metrics = self._calculate_basic_metrics(y, predictions_proba, predictions_binary)

            logger.info("Generating confusion matrix...")
            cm = confusion_matrix(y, predictions_binary)

            evaluation_results = {
                "basic_metrics": metrics, # Renamed key to align with report generation
                "confusion_matrix": cm.tolist(), # Convert to list for JSON serialization
                "prediction_threshold": self.threshold,
                "num_samples_evaluated": len(y)
            }

            # Adding more detailed analyses. These assume the necessary data (like 'sectors')
            # is either available in metadata or can be derived.
            if 'sectors' in metadata:
                 evaluation_results['sector_analysis'] = self._analyze_by_sector(y, predictions_proba, metadata['sectors'])

            # If confidence scores are available, uncomment the following line and ensure they are passed:
            # if 'confidence_scores' in metadata:
            #     evaluation_results['confidence_analysis'] = self._analyze_by_confidence(y, predictions_proba, metadata['confidence_scores'])

            evaluation_results['threshold_analysis'] = self._analyze_thresholds(y, predictions_proba)
            evaluation_results['calibration'] = self._analyze_calibration(y, predictions_proba)
            evaluation_results['risk_distribution'] = self._analyze_risk_distribution(predictions_proba)

            logger.info(f"Evaluation complete. AUC: {metrics.get('auc', 'N/A'):.4f}, F1-Score: {metrics.get('f1_score', 'N/A'):.4f}")
            return evaluation_results

        except Exception as e:
            logger.error(f"An error occurred during model evaluation: {e}", exc_info=True)
            # Return a dictionary indicating failure, with empty metrics
            return {
                "metrics": {},
                "confusion_matrix": [],
                "error": str(e)
            }

    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                                y_pred_binary: np.ndarray) -> Dict:
        """
        Calculates a comprehensive set of classification metrics including AUC,
        Precision-Recall curve metrics, and detailed confusion matrix components.
        """
        metrics = {}

        # Probabilistic metrics
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except ValueError as e:
            metrics['auc'] = 0.0
            logger.warning(f"Could not calculate AUC score, returning 0.0. Reason: {e}")

        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['avg_precision'] = np.mean(precision_curve)
            metrics['auc_pr'] = np.trapz(precision_curve, recall_curve)
        except Exception as e:
            logger.warning(f"Could not calculate Precision-Recall Curve metrics, returning 0.0. Reason: {e}")
            metrics['avg_precision'] = 0.0
            metrics['auc_pr'] = 0.0

        # Binary classification metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
        metrics['precision'] = precision_score(y_true, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred_binary, zero_division=0)
        metrics['f1_score'] = f1_score(y_true, y_pred_binary, zero_division=0)

        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred_binary)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['true_negatives'] = int(tn)
            metrics['false_positives'] = int(fp)
            metrics['false_negatives'] = int(fn)
            metrics['true_positives'] = int(tp)

            # Additional metrics derived from confusion matrix
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        else:
            logger.warning("Confusion matrix not of shape (2,2). Skipping detailed CM metrics.")
            # Ensure these keys are present with default values to avoid KeyError later in report generation
            metrics['true_negatives'] = 0
            metrics['false_positives'] = 0
            metrics['false_negatives'] = 0
            metrics['true_positives'] = 0
            metrics['specificity'] = 0.0
            metrics['sensitivity'] = 0.0
            metrics['positive_predictive_value'] = 0.0
            metrics['negative_predictive_value'] = 0.0

        return metrics

    def _analyze_thresholds(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_analysis = {}

        for threshold in thresholds:
            y_pred_binary = (y_pred_proba > threshold).astype(int)

            # Check if there are at least two unique classes in y_true and y_pred_binary
            # to calculate metrics like precision, recall, f1, etc.
            if len(np.unique(y_true)) < 2 or len(np.unique(y_pred_binary)) < 2:
                logger.warning(f"Skipping metrics for threshold {threshold:.1f}: Not enough unique classes in true or predicted labels.")
                continue

            threshold_analysis[f'threshold_{threshold:.1f}'] = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_binary, zero_division=0)
            }

        # Find optimal threshold based on F1 score
        if not threshold_analysis:
            logger.warning("No valid thresholds generated metrics for optimal threshold analysis.")
            return {
                'optimal_threshold': {
                    'threshold': 0.0,
                    'f1_score': 0.0
                }
            }

        # Collect F1 scores for valid thresholds to find the optimal one
        f1_scores = [threshold_analysis[k]['f1_score'] for k in threshold_analysis if k.startswith('threshold_')]

        if f1_scores:
            optimal_threshold_idx = np.argmax(f1_scores)
            # Re-map the index from the filtered f1_scores back to the original thresholds
            valid_thresholds = [t for t in thresholds if f'threshold_{t:.1f}' in threshold_analysis]
            optimal_threshold = valid_thresholds[optimal_threshold_idx]

            threshold_analysis['optimal_threshold'] = {
                'threshold': float(optimal_threshold),
                'f1_score': float(f1_scores[optimal_threshold_idx])
            }
        else:
            logger.warning("Could not determine optimal threshold as no valid F1 scores were found.")
            threshold_analysis['optimal_threshold'] = {
                'threshold': 0.0,
                'f1_score': 0.0
            }

        return threshold_analysis

    def _analyze_calibration(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Analyze model calibration"""
        try:
            # Calibration curve requires at least two unique classes in y_true
            # and a reasonable spread in y_pred_proba.
            if len(np.unique(y_true)) < 2 or len(np.unique(y_pred_proba)) < 2:
                logger.warning("Skipping calibration analysis: Not enough unique classes in true labels or probabilities.")
                return {
                    'calibration_error': 1.0,
                    'is_well_calibrated': False,
                    'fraction_of_positives': [],
                    'mean_predicted_value': []
                }

            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )

            # Calculate calibration error, handling potential empty arrays
            if len(fraction_of_positives) > 0 and len(mean_predicted_value) > 0:
                calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            else:
                calibration_error = 1.0 # Default to high error if data is insufficient

            return {
                'calibration_error': float(calibration_error),
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'is_well_calibrated': calibration_error < 0.1
            }
        except Exception as e:
            logger.warning(f"Failed to analyze calibration: {e}", exc_info=True)
            return {
                'calibration_error': 1.0,
                'is_well_calibrated': False,
                'fraction_of_positives': [],
                'mean_predicted_value': []
            }

    def _analyze_by_sector(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                          sectors: List[str]) -> Dict:
        """Analyze performance by sector"""
        sector_analysis = {}

        # Ensure sectors list length matches the data lengths
        if len(sectors) != len(y_true) or len(sectors) != len(y_pred_proba):
            logger.warning("Length of sectors list does not match y_true or y_pred_proba. Skipping sector analysis.")
            return {}

        unique_sectors = list(set(sectors))

        for sector in unique_sectors:
            sector_mask = np.array(sectors) == sector

            if sum(sector_mask) < 5:  # Skip sectors with too few samples for meaningful analysis
                logger.info(f"Skipping sector '{sector}' due to insufficient samples ({sum(sector_mask)} < 5).")
                continue

            y_sector = y_true[sector_mask]
            y_pred_sector = y_pred_proba[sector_mask]
            y_pred_binary_sector = (y_pred_sector > self.threshold).astype(int)

            try:
                auc_score = 0.0
                if len(np.unique(y_sector)) > 1: # AUC requires at least two unique classes in the target
                    auc_score = roc_auc_score(y_sector, y_pred_sector)
                else:
                    logger.warning(f"Skipping AUC for sector '{sector}': Not enough unique classes in target.")

                sector_metrics = {
                    'n_samples': int(sum(sector_mask)),
                    'positive_rate': float(y_sector.mean()) if not y_sector.empty else 0.0,
                    'accuracy': accuracy_score(y_sector, y_pred_binary_sector),
                    'precision': precision_score(y_sector, y_pred_binary_sector, zero_division=0),
                    'recall': recall_score(y_sector, y_pred_binary_sector, zero_division=0),
                    'f1_score': f1_score(y_sector, y_pred_binary_sector, zero_division=0),
                    'auc': auc_score
                }

                sector_analysis[sector] = sector_metrics

            except Exception as e:
                logger.warning(f"Failed to analyze sector {sector}: {e}", exc_info=True)
                continue

        return sector_analysis

    def _analyze_risk_distribution(self, y_pred_proba: np.ndarray) -> Dict:
        """Analyze the distribution of predicted risk scores"""
        if y_pred_proba.size == 0:
            logger.warning("No predicted probabilities for risk distribution analysis.")
            return {
                'mean_risk': 0.0, 'median_risk': 0.0, 'std_risk': 0.0,
                'min_risk': 0.0, 'max_risk': 0.0,
                'percentiles': {}, 'risk_buckets': {'counts': {}, 'percentages': {}}
            }

        return {
            'mean_risk': float(np.mean(y_pred_proba)),
            'median_risk': float(np.median(y_pred_proba)),
            'std_risk': float(np.std(y_pred_proba)),
            'min_risk': float(np.min(y_pred_proba)),
            'max_risk': float(np.max(y_pred_proba)),
            'percentiles': {
                '10th': float(np.percentile(y_pred_proba, 10)),
                '25th': float(np.percentile(y_pred_proba, 25)),
                '75th': float(np.percentile(y_pred_proba, 75)),
                '90th': float(np.percentile(y_pred_proba, 90)),
                '95th': float(np.percentile(y_pred_proba, 95)),
                '99th': float(np.percentile(y_pred_proba, 99))
            },
            'risk_buckets': self._create_risk_buckets(y_pred_proba)
        }

    def _create_risk_buckets(self, y_pred_proba: np.ndarray) -> Dict:
        """Create risk buckets for analysis"""
        if y_pred_proba.size == 0:
            return {'counts': {}, 'percentages': {}}

        buckets = {
            'very_low': (y_pred_proba < 0.2).sum(),
            'low': ((y_pred_proba >= 0.2) & (y_pred_proba < 0.4)).sum(),
            'medium': ((y_pred_proba >= 0.4) & (y_pred_proba < 0.6)).sum(),
            'high': ((y_pred_proba >= 0.6) & (y_pred_proba < 0.8)).sum(),
            'very_high': (y_pred_proba >= 0.8).sum()
        }

        total = len(y_pred_proba)
        # Ensure division by zero is handled for percentages
        bucket_percentages = {k: (v / total * 100) if total > 0 else 0.0 for k, v in buckets.items()}

        return {
            'counts': {k: int(v) for k, v in buckets.items()},
            'percentages': bucket_percentages
        }

    def _analyze_by_confidence(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                              confidence_scores: np.ndarray) -> Dict:
        """Analyze performance by model confidence levels"""
        confidence_analysis = {}

        if len(confidence_scores) != len(y_true) or len(confidence_scores) != len(y_pred_proba):
            logger.warning("Length of confidence_scores does not match y_true or y_pred_proba. Skipping confidence analysis.")
            return {}

        # Create confidence buckets
        confidence_buckets = [
            ('low', 0.0, 0.6),
            ('medium', 0.6, 0.8),
            ('high', 0.8, 1.0)
        ]

        for bucket_name, min_conf, max_conf in confidence_buckets:
            mask = (confidence_scores >= min_conf) & (confidence_scores < max_conf)

            if mask.sum() < 5:  # Skip buckets with too few samples
                logger.info(f"Skipping confidence bucket '{bucket_name}' due to insufficient samples ({mask.sum()} < 5).")
                continue

            y_bucket = y_true[mask]
            y_pred_bucket = y_pred_proba[mask]
            y_pred_binary_bucket = (y_pred_bucket > self.threshold).astype(int)

            try:
                auc_score = 0.0
                if len(np.unique(y_bucket)) > 1: # AUC requires at least two unique classes in the target
                    auc_score = roc_auc_score(y_bucket, y_pred_bucket)
                else:
                    logger.warning(f"Skipping AUC for confidence bucket '{bucket_name}': Not enough unique classes in target.")

                bucket_metrics = {
                    'n_samples': int(mask.sum()),
                    'accuracy': accuracy_score(y_bucket, y_pred_binary_bucket),
                    'precision': precision_score(y_bucket, y_pred_binary_bucket, zero_division=0),
                    'recall': recall_score(y_bucket, y_pred_binary_bucket, zero_division=0),
                    'f1_score': f1_score(y_bucket, y_pred_binary_bucket, zero_division=0),
                    'auc': auc_score
                }

                confidence_analysis[bucket_name] = bucket_metrics

            except Exception as e:
                logger.warning(f"Failed to analyze confidence bucket {bucket_name}: {e}", exc_info=True)
                continue

        return confidence_analysis

    def generate_evaluation_report(self, evaluation_results: Dict) -> str:
        """Generate a comprehensive evaluation report based on evaluation_results dictionary."""
        report = []
        report.append("=" * 60)
        report.append("SUPPLY CHAIN RISK MODEL EVALUATION REPORT")
        report.append("=" * 60)

        # Basic metrics
        if 'basic_metrics' in evaluation_results:
            metrics = evaluation_results['basic_metrics']
            report.append("\n📊 BASIC PERFORMANCE METRICS")
            report.append("-" * 30)
            report.append(f"AUC Score: {metrics.get('auc', 0):.4f}")
            report.append(f"Accuracy: {metrics.get('accuracy', 0):.4f}")
            report.append(f"Precision: {metrics.get('precision', 0):.4f}")
            report.append(f"Recall: {metrics.get('recall', 0):.4f}")
            report.append(f"F1 Score: {metrics.get('f1_score', 0):.4f}")

            if 'true_positives' in metrics: # Check if confusion matrix metrics were calculated
                report.append(f"\nConfusion Matrix:")
                report.append(f"True Positives: {metrics['true_positives']}")
                report.append(f"False Positives: {metrics['false_positives']}")
                report.append(f"True Negatives: {metrics['true_negatives']}")
                report.append(f"False Negatives: {metrics['false_negatives']}")

        # Threshold analysis
        if 'threshold_analysis' in evaluation_results:
            threshold_data = evaluation_results['threshold_analysis']
            if 'optimal_threshold' in threshold_data:
                optimal = threshold_data['optimal_threshold']
                report.append(f"\n🎯 OPTIMAL THRESHOLD ANALYSIS")
                report.append("-" * 30)
                report.append(f"Optimal Threshold: {optimal['threshold']:.2f}")
                report.append(f"F1 Score at Optimal: {optimal['f1_score']:.4f}")
            else:
                report.append(f"\n🎯 OPTIMAL THRESHOLD ANALYSIS: No optimal threshold determined.")

        # Calibration
        if 'calibration' in evaluation_results:
            cal_data = evaluation_results['calibration']
            report.append(f"\n📏 MODEL CALIBRATION")
            report.append("-" * 30)
            report.append(f"Calibration Error: {cal_data.get('calibration_error', 1.0):.4f}")
            report.append(f"Well Calibrated: {'Yes' if cal_data.get('is_well_calibrated', False) else 'No'}")

        # Risk distribution
        if 'risk_distribution' in evaluation_results:
            risk_data = evaluation_results['risk_distribution']
            report.append(f"\n📈 RISK SCORE DISTRIBUTION")
            report.append("-" * 30)
            report.append(f"Mean Risk: {risk_data.get('mean_risk', 0):.4f}")
            report.append(f"Median Risk: {risk_data.get('median_risk', 0):.4f}")
            report.append(f"Risk Std Dev: {risk_data.get('std_risk', 0):.4f}")

            if 'risk_buckets' in risk_data and 'percentages' in risk_data['risk_buckets']:
                buckets = risk_data['risk_buckets']['percentages']
                report.append(f"\nRisk Distribution:")
                for bucket, percentage in buckets.items():
                    report.append(f"  {bucket.replace('_', ' ').title()}: {percentage:.1f}%")
            else:
                report.append("  No risk bucket data available.")

        # Sector analysis
        if 'sector_analysis' in evaluation_results:
            sector_data = evaluation_results['sector_analysis']
            if sector_data:
                report.append(f"\n🏭 SECTOR-WISE PERFORMANCE")
                report.append("-" * 30)
                for sector, metrics in sector_data.items():
                    report.append(f"{sector}:")
                    report.append(f"  Samples: {metrics.get('n_samples', 0)}")
                    report.append(f"  AUC: {metrics.get('auc', 0):.4f}")
                    report.append(f"  F1: {metrics.get('f1_score', 0):.4f}")
            else:
                report.append("\n🏭 SECTOR-WISE PERFORMANCE: No sector analysis data available.")


        # Confidence analysis
        if 'confidence_analysis' in evaluation_results:
            conf_data = evaluation_results['confidence_analysis']
            if conf_data:
                report.append(f"\n🎯 CONFIDENCE-BASED PERFORMANCE")
                report.append("-" * 30)
                for conf_level, metrics in conf_data.items():
                    report.append(f"  {conf_level.title()} Confidence:")
                    report.append(f"  Samples: {metrics.get('n_samples', 0)}")
                    report.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                    report.append(f"  F1: {metrics.get('f1_score', 0):.4f}")
            else:
                report.append("\n🎯 CONFIDENCE-BASED PERFORMANCE: No confidence analysis data available.")

        report.append("\n" + "=" * 60)

        return "\n".join(report)

    def save_evaluation_plots(self, y_true: pd.Series, y_pred_proba: np.ndarray,
                             output_dir: str = "evaluation_plots"):
        """Generate and save evaluation plots (ROC, Precision-Recall, Risk Distribution)."""
        import os
        os.makedirs(output_dir, exist_ok=True)

        plt.style.use('default')

        # ROC Curve
        try:
            if len(np.unique(y_true)) < 2:
                logger.warning("Skipping ROC curve plot: Not enough unique classes in true labels.")
            else:
                fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
                auc_score = roc_auc_score(y_true, y_pred_proba)

                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc_score:.3f})')
                plt.plot([0, 1], [0, 1], 'k--', label='Random')
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/roc_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate ROC curve: {e}", exc_info=True)

        # Precision-Recall Curve
        try:
            if len(np.unique(y_true)) < 2:
                logger.warning("Skipping Precision-Recall curve plot: Not enough unique classes in true labels.")
            else:
                precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)

                plt.figure(figsize=(8, 6))
                plt.plot(recall, precision, label='Precision-Recall Curve')
                plt.xlabel('Recall')
                plt.ylabel('Precision')
                plt.title('Precision-Recall Curve')
                plt.legend()
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/precision_recall_curve.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate PR curve: {e}", exc_info=True)

        # Risk Score Distribution
        try:
            if y_pred_proba.size == 0:
                logger.warning("Skipping risk distribution plot: No predicted probabilities.")
            else:
                plt.figure(figsize=(10, 6))
                plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
                plt.xlabel('Predicted Risk Score')
                plt.ylabel('Frequency')
                plt.title('Distribution of Predicted Risk Scores')
                plt.grid(True, alpha=0.3)
                plt.savefig(f"{output_dir}/risk_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate risk distribution plot: {e}", exc_info=True)

        logger.info(f"Evaluation plots saved to {output_dir}/")

    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """Compares multiple model evaluation results to find the best model for key metrics."""
        comparison = {}

        metrics_to_compare = ['auc', 'accuracy', 'precision', 'recall', 'f1_score']

        for metric in metrics_to_compare:
            comparison[metric] = {}
            for model_name, results in model_results.items():
                basic_metrics = results.get('basic_metrics', {}) # Get 'basic_metrics' safely
                comparison[metric][model_name] = basic_metrics.get(metric, 0.0)

        best_models = {}
        for metric, model_scores in comparison.items():
            if model_scores:
                # Check if all scores are 0.0 or if there are no valid scores
                if all(value == 0.0 for value in model_scores.values()) and len(model_scores) > 0:
                     logger.warning(f"All scores for metric '{metric}' are 0.0. Cannot determine best model meaningfully.")
                     best_models[metric] = {'model': 'N/A', 'score': 0.0}
                elif not model_scores: # Check if dictionary is empty after filtering
                    logger.warning(f"No scores available for metric '{metric}'.")
                    best_models[metric] = {'model': 'N/A', 'score': 0.0}
                else:
                    best_model = max(model_scores.items(), key=lambda x: x[1])
                    best_models[metric] = {
                        'model': best_model[0],
                        'score': best_model[1]
                    }
            else:
                best_models[metric] = {'model': 'N/A', 'score': 0.0}

        return {
            'metric_comparison': comparison,
            'best_models': best_models
        }

    def _prepare_lstm_features(self, X: pd.DataFrame) -> np.ndarray:
        """
        Prepares time series features for the LSTM model.
        This is a placeholder and should be replaced with actual time series feature engineering.
        """
        if X.empty:
            return np.array([])
        n_samples = len(X)
        seq_length = 30  # Example sequence length
        n_features = X.shape[1] # Use all columns as features for mock data

        # Create mock time series data shaped (n_samples, seq_length, n_features)
        # This assumes X contains the relevant features to be reshaped.
        # If X has fewer columns than n_features or needs specific feature selection, adjust here.
        # For a simple mock, we'll just take the first n_features columns.
        if n_features < 10: # Ensure we have at least 10 features for the mock
             # Pad with zeros or raise an error if not enough features
             mock_features = np.zeros((n_samples, n_features))
             mock_features[:, :X.shape[1]] = X.values # Fill with actual data
        else:
             mock_features = X.values[:, :n_features] # Take first n_features

        # Reshape to (n_samples, seq_length, n_features)
        # This mock might not be realistic if seq_length > 1 and data is not truly sequential.
        # For proper time series, you'd implement rolling windows, etc.
        # For simplicity, we'll repeat features for the sequence length.
        return np.repeat(mock_features[:, np.newaxis, :], seq_length, axis=1)

