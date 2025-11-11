#!/usr/bin/env python3
"""
Regression-specific reliability and calibration analysis.
Focuses on prediction intervals, residual patterns, and model confidence.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

from analysis_output.calibration.style import styled_subplots

class RegressionReliabilityAnalyzer:
    def __init__(self, results_dir: str = "results/responses"):
        self.results_dir = Path(results_dir)
        self.datasets = ["EI-reg", "V-reg", "SST", "V-A,V-M,V-NYT,V-T"]
        self.models = ["Emobloom-7b", "Emollama-7b", "Emollama-chat-13b", "Emoopt-13b", "Emollama-chat-7b"]
        self.temp_dir = "temp_0.9"
        
        # Define domain ranges
        self.dataset_info = {
            "EI-reg": {"domain": [0, 1], "title": "EI-reg"},
            "V-reg": {"domain": [0, 1], "title": "V-reg"},
            "SST": {"domain": [0, 1], "title": "SST"},
            "V-A,V-M,V-NYT,V-T": {"domain": [-4, 4], "title": "VADER"}
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
    
    def create_reliability_plots(self, save_path: str = "regression_reliability_plots"):
        """Create comprehensive regression reliability analysis plots."""
        os.makedirs(save_path, exist_ok=True)
        
        # Create main reliability plots
        self.create_residual_analysis_plots(save_path)
        
        # Create prediction interval analysis
        self.create_prediction_interval_analysis(save_path)
        
        # Create model consistency analysis
        self.create_model_consistency_analysis(save_path)
        
        # Create reliability summary
        self.create_reliability_summary(save_path)
    
    def create_residual_analysis_plots(self, save_path: str):
        """Create residual analysis plots for regression reliability."""
        fig, axes = styled_subplots(width=16.0, height=12.0, nrows=2, ncols=2)
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Plot residuals vs predictions for each model
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    residuals = true_array - pred_array
                    
                    # Calculate R² score
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((true_array - np.mean(true_array))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Create scatter plot with R² in label
                    markers = ['o', 's', '^', 'D', 'v']
                    marker = markers[self.models.index(model)]
                    
                    ax.scatter(pred_vals, residuals, alpha=0.6, s=20, 
                             label=f'{model} (R²={r_squared:.3f})', marker=marker, edgecolors='white', linewidth=0.3)
            
            # Add zero line
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Set axis properties
            domain_min, domain_max = self.dataset_info[dataset]["domain"]
            ax.set_xlim(domain_min, domain_max)
            ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
            ax.set_ylabel('Residuals (True - Predicted)', fontsize=12, fontweight='bold')
            ax.set_title(f'{self.dataset_info[dataset]["title"]}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8)
        
      
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/residual_analysis.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_prediction_interval_analysis(self, save_path: str):
        """Create prediction interval analysis plots."""
        fig, axes = styled_subplots(width=16.0, height=12.0, nrows=2, ncols=2)
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Calculate and plot prediction intervals for each model
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    residuals = true_array - pred_array
                    
                    # Calculate prediction intervals based on residual distribution
                    residual_std = np.std(residuals)
                    
                    # Create prediction intervals (assuming normal distribution)
                    pred_interval_95 = 1.96 * residual_std
                    pred_interval_68 = 1.0 * residual_std
                    
                    # Calculate R² score
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((true_array - np.mean(true_array))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Plot prediction vs true with intervals
                    markers = ['o', 's', '^', 'D', 'v']
                    marker = markers[self.models.index(model)]
                    
                    # Sort for plotting
                    sort_idx = np.argsort(pred_array)
                    sorted_pred = pred_array[sort_idx]
                    sorted_true = true_array[sort_idx]
                    
                    ax.plot(sorted_pred, sorted_pred, '--', alpha=0.5, color='gray', linewidth=1)
                    ax.fill_between(sorted_pred, 
                                   sorted_pred - pred_interval_95, 
                                   sorted_pred + pred_interval_95, 
                                   alpha=0.1, label=f'{model} 95% PI')
                    ax.scatter(pred_array, true_array, alpha=0.6, s=15, 
                             label=f'{model} (R²={r_squared:.3f})', marker=marker, edgecolors='white', linewidth=0.3)
            
            # Set axis properties
            domain_min, domain_max = self.dataset_info[dataset]["domain"]
            ax.set_xlim(domain_min, domain_max)
            ax.set_ylim(domain_min, domain_max)
            ax.set_xlabel('Predicted Values', fontsize=12, fontweight='bold')
            ax.set_ylabel('True Values', fontsize=12, fontweight='bold')
            ax.set_title(f'Prediction Intervals: {self.dataset_info[dataset]["title"]}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=7)
        
        fig.suptitle('Model Reliability: Prediction Intervals', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/prediction_intervals.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_model_consistency_analysis(self, save_path: str):
        """Create model consistency analysis plots."""
        fig, axes = styled_subplots(width=16.0, height=12.0, nrows=2, ncols=2)
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Calculate consistency metrics for each model
            consistency_metrics = []
            model_names = []
            
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    residuals = true_array - pred_array
                    
                    # Calculate consistency metrics
                    mae = np.mean(np.abs(residuals))
                    rmse = np.sqrt(np.mean(residuals**2))
                    mape = np.mean(np.abs(residuals / (true_array + 1e-8))) * 100
                    
                    # Calculate R²
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((true_array - np.mean(true_array))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Calculate prediction stability (coefficient of variation of residuals)
                    residual_cv = np.std(residuals) / np.mean(np.abs(residuals)) if np.mean(np.abs(residuals)) > 0 else 0
                    
                    consistency_metrics.append({
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'R²': r_squared,
                        'Residual_CV': residual_cv
                    })
                    model_names.append(model)
            
            # Create bar plot of R² scores
            r_squared_scores = [m['R²'] for m in consistency_metrics]
            bars = ax.bar(range(len(model_names)), r_squared_scores, alpha=0.7)
            ax.set_xticks(range(len(model_names)))
            ax.set_xticklabels([name.split('-')[0] for name in model_names], rotation=45)
            ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
            ax.set_title(f'Model Consistency (R²): {self.dataset_info[dataset]["title"]}', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            
            # Color bars based on R² score
            for i, bar in enumerate(bars):
                if r_squared_scores[i] > 0.7:
                    bar.set_color('green')
                elif r_squared_scores[i] > 0.5:
                    bar.set_color('orange')
                else:
                    bar.set_color('red')
        
        fig.suptitle('Model Consistency Analysis (R² Scores)', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.15, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/model_consistency.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_reliability_summary(self, save_path: str):
        """Create reliability summary statistics and plots."""
        summary_data = []
        
        for dataset in self.datasets:
            dataset_data = self.load_dataset_data(dataset)
            
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    residuals = true_array - pred_array
                    
                    # Calculate reliability metrics
                    mae = np.mean(np.abs(residuals))
                    rmse = np.sqrt(np.mean(residuals**2))
                    mape = np.mean(np.abs(residuals / (true_array + 1e-8))) * 100
                    
                    # R²
                    ss_res = np.sum(residuals**2)
                    ss_tot = np.sum((true_array - np.mean(true_array))**2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # Residual statistics
                    residual_std = np.std(residuals)
                    residual_mean = np.mean(residuals)
                    
                    # Prediction interval coverage (95%)
                    pred_interval_95 = 1.96 * residual_std
                    coverage_95 = np.mean((np.abs(residuals) <= pred_interval_95)) * 100
                    
                    summary_data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'R²': r_squared,
                        'MAE': mae,
                        'RMSE': rmse,
                        'MAPE': mape,
                        'Residual_Mean': residual_mean,
                        'Residual_Std': residual_std,
                        'Coverage_95%': coverage_95,
                        'Samples': len(true_array)
                    })
        
        # Create summary DataFrame
        df = pd.DataFrame(summary_data)
        
        # Create comprehensive summary plots
        fig, axes = styled_subplots(width=16.0, height=12.0, nrows=2, ncols=2)
        
        # R² heatmap
        pivot_r2 = df.pivot(index='Model', columns='Dataset', values='R²')
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', center=0.5, 
                   ax=axes[0,0], cbar_kws={'label': 'R² Score'})
        axes[0,0].set_title('Model Reliability (R² Scores)', fontsize=12, fontweight='bold')
        axes[0,0].set_xlabel('Dataset', fontsize=10, fontweight='bold')
        axes[0,0].set_ylabel('Model', fontsize=10, fontweight='bold')
        
        # MAE heatmap
        pivot_mae = df.pivot(index='Model', columns='Dataset', values='MAE')
        sns.heatmap(pivot_mae, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=axes[0,1], cbar_kws={'label': 'MAE'})
        axes[0,1].set_title('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
        axes[0,1].set_xlabel('Dataset', fontsize=10, fontweight='bold')
        axes[0,1].set_ylabel('Model', fontsize=10, fontweight='bold')
        
        # Coverage heatmap
        pivot_coverage = df.pivot(index='Model', columns='Dataset', values='Coverage_95%')
        sns.heatmap(pivot_coverage, annot=True, fmt='.1f', cmap='RdYlGn', center=95, 
                   ax=axes[1,0], cbar_kws={'label': 'Coverage %'})
        axes[1,0].set_title('Prediction Interval Coverage (95%)', fontsize=12, fontweight='bold')
        axes[1,0].set_xlabel('Dataset', fontsize=10, fontweight='bold')
        axes[1,0].set_ylabel('Model', fontsize=10, fontweight='bold')
        
        # Residual standard deviation heatmap
        pivot_res_std = df.pivot(index='Model', columns='Dataset', values='Residual_Std')
        sns.heatmap(pivot_res_std, annot=True, fmt='.4f', cmap='RdYlBu_r', 
                   ax=axes[1,1], cbar_kws={'label': 'Residual Std'})
        axes[1,1].set_title('Residual Standard Deviation', fontsize=12, fontweight='bold')
        axes[1,1].set_xlabel('Dataset', fontsize=10, fontweight='bold')
        axes[1,1].set_ylabel('Model', fontsize=10, fontweight='bold')
        
        plt.suptitle('Regression Model Reliability Summary', fontsize=16, fontweight='bold', y=0.95)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/reliability_summary.png", dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save detailed results
        df.to_csv(f"{save_path}/reliability_metrics.csv", index=False)
        print(f"Regression reliability analysis complete! Results saved to {save_path}")
        print("\nReliability Metrics Summary:")
        print("=" * 80)
        print(df[['Dataset', 'Model', 'R²', 'MAE', 'RMSE', 'Coverage_95%']].to_string(index=False))
        
        return df

def main():
    """Main function to run regression reliability analysis."""
    analyzer = RegressionReliabilityAnalyzer()
    
    print("Starting regression reliability analysis...")
    print("=" * 50)
    
    analyzer.create_reliability_plots()
    
    print("\nRegression reliability analysis complete!")

if __name__ == "__main__":
    main()
