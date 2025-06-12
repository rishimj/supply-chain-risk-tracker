import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import List, Optional, Tuple
import json

@dataclass
class LSTMConfig:
    """Configuration for LSTM model"""
    input_size: int
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = True
    batch_first: bool = True
    
    def to_dict(self):
        """Convert config to dictionary"""
        return {
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'bidirectional': self.bidirectional,
            'batch_first': self.batch_first
        }
    
    @classmethod
    def from_dict(cls, config_dict):
        """Create config from dictionary"""
        return cls(**config_dict)

class LSTMModel(nn.Module):
    """LSTM model for time series prediction"""
    
    def __init__(self, config: LSTMConfig):
        super().__init__()
        self.config = config
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=config.input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout=config.dropout if config.num_layers > 1 else 0,
            bidirectional=config.bidirectional,
            batch_first=config.batch_first
        )
        
        # Calculate output size based on bidirectional setting
        lstm_output_size = config.hidden_size * 2 if config.bidirectional else config.hidden_size
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(lstm_output_size, lstm_output_size // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(lstm_output_size // 2, 1)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Output tensor of shape (batch_size, 1)
        """
        # LSTM forward pass
        lstm_out, _ = self.lstm(x)
        
        # Take the last output of the sequence
        last_output = lstm_out[:, -1, :]
        
        # Pass through fully connected layers
        output = self.fc_layers(last_output)
        
        return output
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Make predictions with the model
        
        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            
        Returns:
            Predictions tensor of shape (batch_size, 1)
        """
        self.eval()
        with torch.no_grad():
            return self.forward(x)
    
    def save(self, path: str):
        """Save model state"""
        # Save model state dict
        torch.save(self.state_dict(), path)
        
        # Save config separately as JSON
        config_path = path + '.config.json'
        with open(config_path, 'w') as f:
            json.dump(self.config.to_dict(), f)
    
    @classmethod
    def load(cls, path: str) -> 'LSTMModel':
        """Load model from saved state"""
        # Load config from JSON
        config_path = path + '.config.json'
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        
        # Create config
        config = LSTMConfig.from_dict(config_dict)
        
        # Create model
        model = cls(config)
        
        # Load state dict
        model.load_state_dict(torch.load(path))
        
        return model 