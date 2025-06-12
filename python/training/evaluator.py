import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_auc_score, roc_curve, precision_recall_curve, confusion_matrix,
    classification_report, accuracy_score, f1_score, precision_score, recall_score
)
from sklearn.calibration import calibration_curve
import logging
from typing import Dict, List, Tuple, Optional, Any
import warnings
import torch
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class ModelEvaluator:
    def __init__(self, config: Dict):
        self.config = config
        self.metrics = config.get('metrics', ['auc', 'precision', 'recall', 'f1_score'])
        self.threshold = config.get('default_threshold', 0.5)
        
    def evaluate_model(self, model: Any, X: pd.DataFrame, y: pd.Series, metadata: Dict) -> Dict:
        """Comprehensive model evaluation"""
        logger.info("Starting comprehensive model evaluation...")
        
        # Prepare time series features for LSTM
        X_lstm = self._prepare_lstm_features(X)
        
        # Get predictions
        predictions = model.predict(
            X_lstm=torch.FloatTensor(X_lstm),
            X_tabular=X.values
        )
        
        # Convert predictions to binary
        y_pred_proba = predictions
        y_pred_binary = (y_pred_proba > self.threshold).astype(int)
        
        evaluation_results = {}
        
        # Basic metrics
        evaluation_results['basic_metrics'] = self._calculate_basic_metrics(y, y_pred_proba, y_pred_binary)
        
        # Threshold analysis
        evaluation_results['threshold_analysis'] = self._analyze_thresholds(y, y_pred_proba)
        
        # Calibration analysis
        evaluation_results['calibration'] = self._analyze_calibration(y, y_pred_proba)
        
        # Feature importance analysis
        if hasattr(model, 'get_feature_importance'):
            evaluation_results['feature_importance'] = model.get_feature_importance()
        
        # Sector-wise analysis
        if 'sectors' in metadata:
            evaluation_results['sector_analysis'] = self._analyze_by_sector(
                y, y_pred_proba, metadata['sectors']
            )
        
        # Risk distribution analysis
        evaluation_results['risk_distribution'] = self._analyze_risk_distribution(y_pred_proba)
        
        # Model stability analysis
        evaluation_results['stability'] = {
            'risk_score_stability': {
                'mean': float(np.mean(y_pred_proba)),
                'std': float(np.std(y_pred_proba)),
                'coefficient_of_variation': float(np.std(y_pred_proba) / np.mean(y_pred_proba)) if np.mean(y_pred_proba) > 0 else 0
            }
        }
        
        # Performance by confidence
        evaluation_results['confidence_analysis'] = self._analyze_by_confidence(
            y, y_pred_proba, predictions.get('confidence', [0.5] * len(y))
        )
        
        logger.info("Model evaluation completed")
        return evaluation_results
    
    def _calculate_basic_metrics(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                                y_pred_binary: np.ndarray) -> Dict:
        """Calculate basic classification metrics"""
        metrics = {}
        
        # Probabilistic metrics
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_proba)
        except:
            metrics['auc'] = 0.0
            
        try:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_pred_proba)
            metrics['avg_precision'] = np.mean(precision_curve)
            metrics['auc_pr'] = np.trapz(precision_curve, recall_curve)
        except:
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
            
            # Additional metrics
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
            metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0
        
        return metrics
    
    def _analyze_thresholds(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Analyze performance across different thresholds"""
        thresholds = np.arange(0.1, 1.0, 0.1)
        threshold_analysis = {}
        
        for threshold in thresholds:
            y_pred_binary = (y_pred_proba > threshold).astype(int)
            
            threshold_analysis[f'threshold_{threshold:.1f}'] = {
                'accuracy': accuracy_score(y_true, y_pred_binary),
                'precision': precision_score(y_true, y_pred_binary, zero_division=0),
                'recall': recall_score(y_true, y_pred_binary, zero_division=0),
                'f1_score': f1_score(y_true, y_pred_binary, zero_division=0)
            }
        
        # Find optimal threshold based on F1 score
        f1_scores = [threshold_analysis[f'threshold_{t:.1f}']['f1_score'] for t in thresholds]
        optimal_threshold_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_threshold_idx]
        
        threshold_analysis['optimal_threshold'] = {
            'threshold': float(optimal_threshold),
            'f1_score': float(f1_scores[optimal_threshold_idx])
        }
        
        return threshold_analysis
    
    def _analyze_calibration(self, y_true: pd.Series, y_pred_proba: np.ndarray) -> Dict:
        """Analyze model calibration"""
        try:
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_true, y_pred_proba, n_bins=10
            )
            
            # Calculate calibration error
            calibration_error = np.mean(np.abs(fraction_of_positives - mean_predicted_value))
            
            return {
                'calibration_error': float(calibration_error),
                'fraction_of_positives': fraction_of_positives.tolist(),
                'mean_predicted_value': mean_predicted_value.tolist(),
                'is_well_calibrated': calibration_error < 0.1
            }
        except:
            return {
                'calibration_error': 1.0,
                'is_well_calibrated': False
            }
    
    def _analyze_by_sector(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                          sectors: List[str]) -> Dict:
        """Analyze performance by sector"""
        sector_analysis = {}
        
        unique_sectors = list(set(sectors))
        
        for sector in unique_sectors:
            sector_mask = [s == sector for s in sectors]
            
            if sum(sector_mask) < 5:  # Skip sectors with too few samples
                continue
                
            y_sector = y_true[sector_mask]
            y_pred_sector = y_pred_proba[sector_mask]
            y_pred_binary_sector = (y_pred_sector > self.threshold).astype(int)
            
            try:
                sector_metrics = {
                    'n_samples': int(sum(sector_mask)),
                    'positive_rate': float(y_sector.mean()),
                    'accuracy': accuracy_score(y_sector, y_pred_binary_sector),
                    'precision': precision_score(y_sector, y_pred_binary_sector, zero_division=0),
                    'recall': recall_score(y_sector, y_pred_binary_sector, zero_division=0),
                    'f1_score': f1_score(y_sector, y_pred_binary_sector, zero_division=0)
                }
                
                if len(np.unique(y_sector)) > 1:
                    sector_metrics['auc'] = roc_auc_score(y_sector, y_pred_sector)
                else:
                    sector_metrics['auc'] = 0.0
                    
                sector_analysis[sector] = sector_metrics
                
            except Exception as e:
                logger.warning(f"Failed to analyze sector {sector}: {e}")
                continue
        
        return sector_analysis
    
    def _analyze_risk_distribution(self, y_pred_proba: np.ndarray) -> Dict:
        """Analyze the distribution of predicted risk scores"""
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
        buckets = {
            'very_low': (y_pred_proba < 0.2).sum(),
            'low': ((y_pred_proba >= 0.2) & (y_pred_proba < 0.4)).sum(),
            'medium': ((y_pred_proba >= 0.4) & (y_pred_proba < 0.6)).sum(),
            'high': ((y_pred_proba >= 0.6) & (y_pred_proba < 0.8)).sum(),
            'very_high': (y_pred_proba >= 0.8).sum()
        }
        
        total = len(y_pred_proba)
        bucket_percentages = {k: (v / total * 100) for k, v in buckets.items()}
        
        return {
            'counts': {k: int(v) for k, v in buckets.items()},
            'percentages': bucket_percentages
        }
    
    def _analyze_by_confidence(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                              confidence_scores: np.ndarray) -> Dict:
        """Analyze performance by model confidence levels"""
        confidence_analysis = {}
        
        # Create confidence buckets
        confidence_buckets = [
            ('low', 0.0, 0.6),
            ('medium', 0.6, 0.8),
            ('high', 0.8, 1.0)
        ]
        
        for bucket_name, min_conf, max_conf in confidence_buckets:
            mask = (confidence_scores >= min_conf) & (confidence_scores < max_conf)
            
            if mask.sum() < 5:  # Skip buckets with too few samples
                continue
            
            y_bucket = y_true[mask]
            y_pred_bucket = y_pred_proba[mask]
            y_pred_binary_bucket = (y_pred_bucket > self.threshold).astype(int)
            
            try:
                bucket_metrics = {
                    'n_samples': int(mask.sum()),
                    'accuracy': accuracy_score(y_bucket, y_pred_binary_bucket),
                    'precision': precision_score(y_bucket, y_pred_binary_bucket, zero_division=0),
                    'recall': recall_score(y_bucket, y_pred_binary_bucket, zero_division=0),
                    'f1_score': f1_score(y_bucket, y_pred_binary_bucket, zero_division=0)
                }
                
                if len(np.unique(y_bucket)) > 1:
                    bucket_metrics['auc'] = roc_auc_score(y_bucket, y_pred_bucket)
                else:
                    bucket_metrics['auc'] = 0.0
                
                confidence_analysis[bucket_name] = bucket_metrics
                
            except Exception as e:
                logger.warning(f"Failed to analyze confidence bucket {bucket_name}: {e}")
                continue
        
        return confidence_analysis
    
    def generate_evaluation_report(self, evaluation_results: Dict) -> str:
        """Generate a comprehensive evaluation report"""
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
            
            if 'true_positives' in metrics:
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
        
        # Confidence analysis
        if 'confidence_analysis' in evaluation_results:
            conf_data = evaluation_results['confidence_analysis']
            if conf_data:
                report.append(f"\n🎯 CONFIDENCE-BASED PERFORMANCE")
                report.append("-" * 30)
                for conf_level, metrics in conf_data.items():
                    report.append(f"{conf_level.title()} Confidence:")
                    report.append(f"  Samples: {metrics.get('n_samples', 0)}")
                    report.append(f"  Accuracy: {metrics.get('accuracy', 0):.4f}")
                    report.append(f"  F1: {metrics.get('f1_score', 0):.4f}")
        
        report.append("\n" + "=" * 60)
        
        return "\n".join(report)
    
    def save_evaluation_plots(self, y_true: pd.Series, y_pred_proba: np.ndarray, 
                             output_dir: str = "evaluation_plots"):
        """Generate and save evaluation plots"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        plt.style.use('default')
        
        # ROC Curve
        try:
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
            logger.warning(f"Failed to generate ROC curve: {e}")
        
        # Precision-Recall Curve
        try:
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
            logger.warning(f"Failed to generate PR curve: {e}")
        
        # Risk Score Distribution
        try:
            plt.figure(figsize=(10, 6))
            plt.hist(y_pred_proba, bins=50, alpha=0.7, edgecolor='black')
            plt.xlabel('Predicted Risk Score')
            plt.ylabel('Frequency')
            plt.title('Distribution of Predicted Risk Scores')
            plt.grid(True, alpha=0.3)
            plt.savefig(f"{output_dir}/risk_distribution.png", dpi=300, bbox_inches='tight')
            plt.close()
        except Exception as e:
            logger.warning(f"Failed to generate risk distribution plot: {e}")
        
        logger.info(f"Evaluation plots saved to {output_dir}/")
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict:
        """Compare multiple model evaluation results"""
        comparison = {}
        
        # Extract key metrics for comparison
        metrics_to_compare = ['auc', 'accuracy', 'precision', 'recall', 'f1_score']
        
        for metric in metrics_to_compare:
            comparison[metric] = {}
            for model_name, results in model_results.items():
                basic_metrics = results.get('basic_metrics', {})
                comparison[metric][model_name] = basic_metrics.get(metric, 0.0)
        
        # Find best model for each metric
        best_models = {}
        for metric, model_scores in comparison.items():
            if model_scores:
                best_model = max(model_scores.items(), key=lambda x: x[1])
                best_models[metric] = {
                    'model': best_model[0],
                    'score': best_model[1]
                }
        
        return {
            'metric_comparison': comparison,
            'best_models': best_models
        }
    
    def _prepare_lstm_features(self, X: pd.DataFrame) -> np.ndarray:
        """Prepare time series features for LSTM model"""
        # In a real implementation, this would reshape the data properly
        # For now, create a simple mock time series
        n_samples = len(X)
        seq_length = 30  # 30 time steps
        n_features = 10  # 10 features per time step
        
        # Create mock time series data
        return np.random.randn(n_samples, seq_length, n_features) 