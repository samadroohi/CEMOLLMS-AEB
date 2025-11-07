#!/usr/bin/env python3
"""
Create a comprehensive presentation plot showing prediction vs true values for all datasets.
"""

import json
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Set style for presentation
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("Set2")

class PresentationPlotter:
    def __init__(self, results_dir: str = "results/responses"):
        self.results_dir = Path(results_dir)
        self.datasets = ["EI-reg", "V-reg", "SST", "V-A,V-M,V-NYT,V-T"]
        self.models = ["Emobloom-7b", "Emollama-7b", "Emollama-chat-13b", "Emollama-chat-7b", "Emoopt-13b"]
        self.temp_dir = "temp_0.9"
        
        # Define domain ranges and colors
        self.dataset_info = {
            "EI-reg": {"domain": [0, 1], "title": "EI-reg", "color": "#1f77b4"},
            "V-reg": {"domain": [0, 1], "title": "V-reg", "color": "#ff7f0e"},
            "SST": {"domain": [0, 1], "title": "SST", "color": "#2ca02c"},
            "V-A,V-M,V-NYT,V-T": {"domain": [-4, 4], "title": "V-A,V-M,V-NYT,V-T", "color": "#d62728"}
        }
        
    def clean_prediction_value(self, pred_str: str) -> Optional[float]:
        """Clean and parse prediction values, handling various formats."""
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
                    # Clip predictions to domain
                    clipped_predictions = self.clip_predictions_to_domain(predictions, dataset)
                    dataset_data[model] = list(zip(true_values, clipped_predictions))
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return dataset_data
    
    def create_presentation_plot(self, save_path: str = "presentation_plots"):
        """Create a comprehensive presentation plot."""
        os.makedirs(save_path, exist_ok=True)
        
        # Create figure with subplots - more compact for presentation
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Process each dataset
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Create scatter plot
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    # Plot with different markers for each model
                    markers = ['o', 's', '^', 'D', 'v']
                    marker = markers[self.models.index(model)]
                    
                    ax.scatter(true_vals, pred_vals, alpha=0.6, s=25, 
                             label=model, marker=marker, edgecolors='white', linewidth=0.3)
            
            # Add perfect prediction line
            domain_min, domain_max = self.dataset_info[dataset]["domain"]
            ax.plot([domain_min, domain_max], [domain_min, domain_max], 
                   'r--', alpha=0.8, linewidth=2, label='Perfect Prediction')
            
            # Set axis properties
            ax.set_xlim(domain_min, domain_max)
            ax.set_ylim(domain_min, domain_max)
            ax.set_title(self.dataset_info[dataset]["title"], fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            
            # Adjust tick labels to prevent overlap
            ax.tick_params(axis='x', labelsize=10, pad=8)
            ax.tick_params(axis='y', labelsize=10, pad=8)
            
            # Calculate and display R² scores in a cleaner way
            r_squared_data = []
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    # Calculate R²
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    ss_res = np.sum((true_array - pred_array) ** 2)
                    ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    r_squared_data.append((model, r_squared))
            
            # Sort by R² score for better readability
            r_squared_data.sort(key=lambda x: x[1], reverse=True)
            
            # Add R² scores in a more compact format
            r_squared_text = "R²:\n" + "\n".join([f"{model.split('-')[0]}: {r2:.3f}" for model, r2 in r_squared_data])
            ax.text(0.02, 0.98, r_squared_text, transform=ax.transAxes, 
                   verticalalignment='top', fontsize=7, 
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.9, edgecolor='gray'))
            
            # Set aspect ratio to be square
            ax.set_aspect('equal', adjustable='box')
        
        # Add shared axis labels
        fig.text(0.5, 0.02, 'True Values', ha='center', va='center', fontsize=14, fontweight='bold')
        fig.text(0.02, 0.5, 'Predictions', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        
        # Add overall title
        fig.suptitle('Model Performance: Predictions vs True Values', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add legend for all subplots (only once)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), 
                  ncol=6, fontsize=9, frameon=True, fancybox=True, shadow=True)
        
        # Adjust layout to prevent overlaps - more compact for presentation
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        # Save plot
        plt.savefig(f"{save_path}/presentation_predictions_vs_true_clean.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Create a second plot focusing on residuals
        self.create_residual_presentation_plot(save_path)
        
        # Create a third plot with model comparison
        self.create_model_comparison_plot(save_path)
    
    def create_residual_presentation_plot(self, save_path: str):
        """Create residual analysis presentation plot."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        for idx, dataset in enumerate(self.datasets):
            row, col = idx // 2, idx % 2
            ax = axes[row, col]
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                continue
            
            # Calculate residuals for each model
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    residuals = [true - pred for true, pred in zip(true_vals, pred_vals)]
                    
                    # Plot residuals
                    markers = ['o', 's', '^', 'D', 'v']
                    marker = markers[self.models.index(model)]
                    
                    ax.scatter(true_vals, residuals, alpha=0.6, s=25, 
                             label=model, marker=marker, edgecolors='white', linewidth=0.3)
            
            # Add zero line
            domain_min, domain_max = self.dataset_info[dataset]["domain"]
            ax.axhline(y=0, color='red', linestyle='--', alpha=0.8, linewidth=2)
            
            # Set axis properties
            ax.set_xlim(domain_min, domain_max)
            ax.set_title(f'{self.dataset_info[dataset]["title"]} - Residuals', 
                        fontsize=12, fontweight='bold', pad=15)
            ax.grid(True, alpha=0.3)
            
            # Adjust tick labels to prevent overlap
            ax.tick_params(axis='x', labelsize=10, pad=8)
            ax.tick_params(axis='y', labelsize=10, pad=8)
        
        # Add shared axis labels
        fig.text(0.5, 0.02, 'True Values', ha='center', va='center', fontsize=14, fontweight='bold')
        fig.text(0.02, 0.5, 'Residuals (True - Predicted)', ha='center', va='center', rotation='vertical', fontsize=14, fontweight='bold')
        
        fig.suptitle('Residual Analysis: Model Errors', 
                    fontsize=16, fontweight='bold', y=0.95)
        
        # Add legend
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, labels, loc='center', bbox_to_anchor=(0.5, 0.05), 
                  ncol=6, fontsize=9, frameon=True, fancybox=True, shadow=True)
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.90, bottom=0.12, left=0.08, right=0.95, hspace=0.25, wspace=0.15)
        
        plt.savefig(f"{save_path}/presentation_residuals_clean.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
    
    def create_model_comparison_plot(self, save_path: str):
        """Create model performance comparison plot."""
        # Collect performance data
        performance_data = []
        
        for dataset in self.datasets:
            dataset_data = self.load_dataset_data(dataset)
            
            for model in self.models:
                if model in dataset_data:
                    model_data = dataset_data[model]
                    true_vals, pred_vals = zip(*model_data)
                    
                    # Calculate metrics
                    true_array = np.array(true_vals)
                    pred_array = np.array(pred_vals)
                    
                    # R²
                    ss_res = np.sum((true_array - pred_array) ** 2)
                    ss_tot = np.sum((true_array - np.mean(true_array)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
                    
                    # MAE
                    mae = np.mean(np.abs(true_array - pred_array))
                    
                    performance_data.append({
                        'Dataset': self.dataset_info[dataset]["title"],
                        'Model': model,
                        'R_squared': r_squared,
                        'MAE': mae
                    })
        
        df = pd.DataFrame(performance_data)
        
        # Create subplots
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
        
        # R² comparison
        pivot_r2 = df.pivot(index='Model', columns='Dataset', values='R_squared')
        sns.heatmap(pivot_r2, annot=True, fmt='.3f', cmap='RdYlGn', center=0, 
                   ax=axes[0], cbar_kws={'label': 'R² Score'})
        axes[0].set_title('Model Performance Comparison (R² Scores)', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Dataset', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Model', fontsize=12, fontweight='bold')
        
        # MAE comparison
        pivot_mae = df.pivot(index='Model', columns='Dataset', values='MAE')
        sns.heatmap(pivot_mae, annot=True, fmt='.3f', cmap='RdYlBu_r', 
                   ax=axes[1], cbar_kws={'label': 'MAE'})
        axes[1].set_title('Model Performance Comparison (MAE)', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Dataset', fontsize=12, fontweight='bold')
        axes[1].set_ylabel('Model', fontsize=12, fontweight='bold')
        
        plt.suptitle('Comprehensive Model Performance Comparison', 
                    fontsize=18, fontweight='bold', y=1.02)
        
        plt.tight_layout()
        plt.savefig(f"{save_path}/presentation_model_comparison.png", 
                   dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        
        # Save performance data
        df.to_csv(f"{save_path}/presentation_performance_data.csv", index=False)
        print(f"Performance data saved to: {save_path}/presentation_performance_data.csv")

def main():
    """Main function to create presentation plots."""
    plotter = PresentationPlotter()
    
    print("Creating presentation plots...")
    print("=" * 50)
    
    plotter.create_presentation_plot()
    
    print("\nPresentation plots created successfully!")
    print("Check the 'presentation_plots' directory for results.")

if __name__ == "__main__":
    main()
