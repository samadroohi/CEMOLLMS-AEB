#!/usr/bin/env python3
"""
Comprehensive calibration analysis for emotion analysis models.
Shows how well-calibrated models are for each dataset.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
import warnings
warnings.filterwarnings('ignore')

# Set style for presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class CalibrationAnalyzer:
    def __init__(self, results_dir: str = "results/responses"):
        self.results_dir = Path(results_dir)
        self.datasets = ["EI-reg", "V-reg", "SST", "V-A,V-M,V-NYT,V-T"]
        self.models = ["Emobloom-7b", "Emollama-7b", "Emollama-chat-13b", "Emollama-chat-7b", "Emoopt-13b"]
        self.temp_dir = "temp_0.9"
        
        # Define domain ranges
        self.dataset_info = {
            "EI-reg": {"domain": [0, 1], "title": "EI-reg"},
            "V-reg": {"domain": [0, 1], "title": "V-reg"},
            "SST": {"domain": [0, 1], "title": "SST"},
            "V-A,V-M,V-NYT,V-T": {"domain": [-4, 4], "title": "V-A,V-M,V-NYT,V-T"}
        }
        
    def clean_prediction_value(self, pred_str: str) -> Optional[float]:
        """Clean and parse prediction values."""
        if not pred_str or pred_str == "null":
            return None
            
        cleaned = str(pred_str).strip()
        import re
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None
    
    def clip_predictions_to_domain(self, predictions: List[float], dataset: str) -> List[float]:
        """Clip predictions to domain range."""
        domain_min, domain_max = self.dataset_info[dataset]["domain"]
        clipped_predictions = []
        
        for pred in predictions:
            if pred < domain_min:
                clipped_predictions.append(domain_min)
            elif pred > domain_max:
                clipped_predictions.append(domain_max)
            else:
                clipped_predictions.append(pred)
                
        return clipped_predictions
    
    def load_dataset_data(self, dataset: str) -> Dict[str, List[Tuple[float, float]]]:
        """Load data for a specific dataset across all models."""
        dataset_data = {}
        
        for model in self.models:
            file_path = self.results_dir / dataset / self.temp_dir / f"{model}.json"
            
            if not file_path.exists():
                continue
                
            true_values = []
            predictions = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            
                            try:
                                true_val = float(data['true_value'])
                            except (ValueError, KeyError):
                                continue
                                
                            pred_val = self.clean_prediction_value(data.get('prediction'))
                            if pred_val is None:
                                continue
                                
                            true_values.append(true_val)
                            predictions.append(pred_val)
                
                if true_values and predictions:
                    clipped_predictions = self.clip_predictions_to_domain(predictions, dataset)
                    dataset_data[model] = list(zip(true_values, clipped_predictions))
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return dataset_data
    
    def create_calibration_plots(self, save_path: str = "calibration_plots"):
        """Create comprehensive calibration analysis plots."""
        os.makedirs(save_path, exist_ok=True)
        
        # Create main calibration plot
        self.create_reliability_diagrams(save_path)
        
        # Create calibration error analysis
        self.create_calibration_error_analysis(save_path)
        
        # Create confidence vs accuracy plots
        self.create_confidence_accuracy_plots(save_path)
        
        # Create calibration summary
        self.create_calibration_summary(save_path)
    
    def create_reliability_diagrams(self, save_path: str):
        """Create reliability diagrams (calibration curves) for each dataset."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Plot calibration curves for each model
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    # Convert to numpy arrays
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    
                    # For regression calibration, we need to bin the data
                    # Create bins based on prediction values
                    n_bins = 10
                    bin_boundaries = np.linspace(0, 1, n_bins + 1) if dataset != "V-A,V-M,V-NYT,V-T" else np.linspace(-4, 4, n_bins + 1)
                    
                    bin_lowers = bin_boundaries[:-1]
                    bin_uppers = bin_boundaries[1:]
                    
                    bin_centers = []
                    bin_accuracies = []
                    bin_confidences = []
                    
                    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
                        in_bin = (pred_array >= bin_lower) & (pred_array < bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            bin_centers.append((bin_lower + bin_upper) / 2)
                            bin_accuracies.append(true_array[in_bin].mean())
                            bin_confidences.append(pred_array[in_bin].mean())
                    
                    if bin_centers:
                        # Plot calibration curve
                        markers = ['o', 's', '^', 'D', 'v']
                        marker = markers[self.models.index(model)]
                        
                        ax.plot(bin_confidences, bin_accuracies, marker=marker, 
                               label=model, linewidth=2, markersize=6, alpha=0.8)
            
            # Add perfect calibration line
            domain_min, domain_max = self.dataset_info[dataset]["domain"]
            ax.plot([domain_min, domain_max], [domain_min, domain_max], 
                   'r--', alpha=0.8, linewidth=2, label='Perfect Calibration')
            
            # Set axis properties
            ax.set_xlim(domain_min, domain_max)
            ax.set_ylim(domain_min, domain_max)
            ax.set_xlabel('Mean Predicted Value', fontsize=12, fontweight='bold')
            ax.set_ylabel('Mean True Value', fontsize=12, fontweight='bold')
            ax.set_title(f'Reliability Diagram: {self.dataset_info[dataset]["title"]}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        # Add shared axis labels
        fig.text(0.5, 0.02, 'Mean Predicted Value', ha='center', va='center', fontsize=14, fontweight='bold')
        fig.text(0.02, 0.5, 'Mean True Value', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        
        fig.suptitle('Model Calibration: Reliability Diagrams', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/reliability_diagrams.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_calibration_error_analysis(self, save_path: str):
        """Create calibration error analysis plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Calculate calibration errors for each model
            calibration_errors = []
            model_names = []
            
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    
                    # Calculate Expected Calibration Error (ECE)
                    n_bins = 10
                    bin_boundaries = np.linspace(0, 1, n_bins + 1) if dataset != "V-A,V-M,V-NYT,V-T" else np.linspace(-4, 4, n_bins + 1)
                    
                    ece = 0
                    for i in range(n_bins):
                        bin_lower = bin_boundaries[i]
                        bin_upper = bin_boundaries[i + 1]
                        
                        in_bin = (pred_array >= bin_lower) & (pred_array < bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = true_array[in_bin].mean()
                            avg_confidence_in_bin = pred_array[in_bin].mean()
                            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
                    
                    calibration_errors.append(ece)
                    model_names.append(model)
            
            # Create bar plot of calibration errors
            bars = ax.bar(range(len(model_names)), calibration_errors, alpha=0.7)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([name.split('-')[0] for name in model_names], rotation=45)
            ax.set_ylabel('Expected Calibration Error (ECE)', fontsize=12, fontweight='bold')
            ax.set_title(f'Calibration Error: {self.dataset_info[dataset]["title"]}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            
            # Color bars based on error level
            for i, bar in enumerate(bars):
                if calibration_errors[i] < 0.05:
                    bar.set_color('green')
                elif calibration_errors[i] < 0.1:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        fig.suptitle('Model Calibration Error Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/calibration_errors.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_confidence_accuracy_plots(self, save_path: str):
        """Create confidence vs accuracy plots."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Plot confidence vs accuracy for each model
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    
                    # Calculate confidence (prediction magnitude) and accuracy (how close to true value)
                    confidence = np.abs(pred_array)
                    accuracy = 1 - np.abs(true_array - pred_array) / (np.max(true_array) - np.min(true_array))
                    
                    # Create scatter plot
                    markers = ['o', 's', '^', 'D', 'v']
                    marker = markers[self.models.index(model)]
                    
                    ax.scatter(confidence, accuracy, alpha=0.6, s=20, 
                             label=model, marker=marker, edgecolors='white', linewidth=0.3)
            
            ax.set_xlabel('Confidence (|Prediction|)', fontsize=12, fontweight='bold')
            ax.set_ylabel('Accuracy (1 - |Error|/Range)', fontsize=12, fontweight='bold')
            ax.set_title(f'Confidence vs Accuracy: {self.dataset_info[dataset]["title"]}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
        fig.suptitle('Model Confidence vs Accuracy Analysis', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/confidence_accuracy.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_calibration_summary(self, save_path: str):
        """Create calibration summary statistics."""
        summary_data = []
        
        for dataset in self.datasets:
            dataset_data = self.load_dataset_data(dataset)
            
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    
                    # Calculate calibration metrics
                    n_bins = 10
                    bin_boundaries = np.linspace(0, 1, n_bins + 1) if dataset != "V-A,V-M,V-NYT,V-T" else np.linspace(-4, 4, n_bins + 1)
                    
                    ece = 0
                    mce = 0
                    for i in range(n_bins):
                        bin_lower = bin_boundaries[i]
                        bin_upper = bin_boundaries[i + 1]
                        
                        in_bin = (pred_array >= bin_lower) & (pred_array < bin_upper)
                        prop_in_bin = in_bin.mean()
                        
                        if prop_in_bin > 0:
                            accuracy_in_bin = true_array[in_bin].mean()
                            avg_confidence_in_bin = pred_array[in_bin].mean()
                            bin_error = np.abs(avg_confidence_in_bin - accuracy_in_bin)
                            
                            ece += bin_error * prop_in_bin
                            mce = max(mce, bin_error)
                    
                    # Calculate Brier Score (for regression, we use MSE)
                    brier_score = np.mean((true_array - pred_array) ** 2)
                    
                    summary_data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'ECE': ece,
                        'MCE': mce,
                        'Brier_Score': brier_score,
                        'Samples': len(true_array)
                    })
        
        # Create summary DataFrame
        df = pd.DataFrame(summary_data)
        
        # Create heatmap of calibration errors
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # ECE heatmap
        pivot_ece = df.pivot(index='Model', columns='Dataset', values='ECE')
        sns.heatmap(pivot_ece, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   ax=axes[0], cbar_kws={'label': 'Expected Calibration Error'})
        axes[0].set_title('Expected Calibration Error (ECE) by Model and Dataset', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dataset', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Model', fontsize=12, fontweight='bold')
        
        # MCE heatmap
        pivot_mce = df.pivot(index='Model', columns='Dataset', values='MCE')
        sns.heatmap(pivot_mce, annot=True, fmt='.4f', cmap='RdYlGn_r', 
                   ax=axes[1], cbar_kws={'label': 'Maximum Calibration Error'})
        axes[1].set_title('Maximum Calibration Error (MCE) by Model and Dataset', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dataset', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.suptitle('Model Calibration Summary', fontsize=16, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/calibration_summary.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save detailed results
        df.to_csv(f"{save_path}/calibration_metrics.csv", index=False)
        print(f"Calibration analysis complete! Results saved to {save_path}")
        print("\nCalibration Metrics Summary:")
        print("=" * 60)
        print(df[['Dataset', 'Model', 'ECE', 'MCE', 'Brier_Score']].to_string(index=False))
        
        return df

def main():
    """Main function to run calibration analysis."""
    analyzer = CalibrationAnalyzer()
    
    print("Starting calibration analysis...")
    print("=" * 50)
    
    analyzer.create_calibration_plots()
    
    print("\nCalibration analysis complete!")

if __name__ == "__main__":
    main()
