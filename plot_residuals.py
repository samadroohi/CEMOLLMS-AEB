#!/usr/bin/env python3
"""
Script to plot residuals between true values and predictions for emotion analysis datasets.
Supports datasets: EI-reg, V-reg, SST, and V-A,V-M,V-NYT,V-T
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

from analysis_output.calibration.style import styled_subplots

class ResidualAnalyzer:
    def __init__(self, results_dir: str = "results/responses"):
        self.results_dir = Path(results_dir)
        self.datasets = ["EI-reg", "V-reg", "SST", "V-A,V-M,V-NYT,V-T"]
        self.models = ["Emobloom-7b", "Emollama-7b", "Emollama-chat-13b", "Emollama-chat-7b", "Emoopt-13b"]
        self.temp_dir = "temp_0.9"
        
        # Define domain ranges for consistent plotting
        self.dataset_domains = {
            "EI-reg": [0, 1],
            "V-reg": [0, 1], 
            "SST": [0, 1],
            "V-A,V-M,V-NYT,V-T": [-4, 4]
        }
        
    def clean_prediction_value(self, pred_str: str) -> Optional[float]:
        """Clean and parse prediction values, handling various formats."""
        if not pred_str or pred_str == "null":
            return None
            
        # Remove extra whitespace and common prefixes
        cleaned = str(pred_str).strip()
        
        # Handle cases where prediction might have extra text
        # Extract the first number found
        import re
        numbers = re.findall(r'-?\d+\.?\d*', cleaned)
        if numbers:
            try:
                return float(numbers[0])
            except ValueError:
                return None
        return None
    
    def clip_predictions_to_domain(self, predictions: List[float], dataset: str) -> Tuple[List[float], int]:
        """Clip predictions to domain range and count violations."""
        domain_min, domain_max = self.dataset_domains[dataset]
        clipped_predictions = []
        violations = 0
        
        for pred in predictions:
            if pred < domain_min:
                clipped_predictions.append(domain_min)
                violations += 1
            elif pred > domain_max:
                clipped_predictions.append(domain_max)
                violations += 1
            else:
                clipped_predictions.append(pred)
                
        return clipped_predictions, violations
    
    def load_dataset_data(self, dataset: str) -> Dict[str, List[Tuple[float, float]]]:
        """Load data for a specific dataset across all models."""
        dataset_data = {}
        
        for model in self.models:
            file_path = self.results_dir / dataset / self.temp_dir / f"{model}.json"
            
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
                
            true_values = []
            predictions = []
            raw_predictions = []
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            
                            # Parse true value
                            try:
                                true_val = float(data['true_value'])
                            except (ValueError, KeyError):
                                continue
                                
                            # Parse and clean prediction
                            pred_val = self.clean_prediction_value(data.get('prediction'))
                            if pred_val is None:
                                continue
                                
                            true_values.append(true_val)
                            raw_predictions.append(pred_val)
                
                if true_values and raw_predictions:
                    # Clip predictions to domain and count violations
                    clipped_predictions, violations = self.clip_predictions_to_domain(raw_predictions, dataset)
                    dataset_data[model] = list(zip(true_values, clipped_predictions))
                    
                    # Report domain violations
                    violation_pct = (violations / len(raw_predictions)) * 100
                    if violations > 0:
                        print(f"Loaded {len(true_values)} samples for {dataset} - {model} (⚠️  {violations} domain violations, {violation_pct:.1f}%)")
                    else:
                        print(f"Loaded {len(true_values)} samples for {dataset} - {model}")
                else:
                    print(f"No valid data found for {dataset} - {model}")
                    
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return dataset_data
    
    def calculate_residuals(self, true_values: List[float], predictions: List[float]) -> List[float]:
        """Calculate residuals (true - predicted)."""
        return [true - pred for true, pred in zip(true_values, predictions)]
    
    def plot_residuals_by_dataset(self, save_path: str = "residual_plots"):
        """Create residual plots for each dataset."""
        os.makedirs(save_path, exist_ok=True)
        
        for dataset in self.datasets:
            print(f"\nProcessing dataset: {dataset}")
            dataset_data = self.load_dataset_data(dataset)
            
            if not dataset_data:
                print(f"No data found for {dataset}")
                continue
            
            # Create subplots for this dataset
            n_models = len(dataset_data)
            if n_models == 0:
                continue
                
            fig, axes = styled_subplots(width=18.0, height=12.0, nrows=2, ncols=3)
            axes = axes.flatten()
            
            # Plot residuals for each model
            for idx, (model, data) in enumerate(dataset_data.items()):
                if idx >= len(axes):
                    break
                    
                true_vals, pred_vals = zip(*data)
                residuals = self.calculate_residuals(true_vals, pred_vals)
                
                ax = axes[idx]
                
                # Scatter plot of residuals vs true values
                ax.scatter(true_vals, residuals, alpha=0.6, s=20)
                ax.axhline(y=0, color='red', linestyle='--', alpha=0.7)
                ax.set_xlabel('True Values')
                ax.set_ylabel('Residuals (True - Predicted)')
                ax.set_title(f'{model}\nResiduals vs True Values')
                ax.grid(True, alpha=0.3)
                
                # Set consistent axis ranges based on dataset domain
                domain_min, domain_max = self.dataset_domains[dataset]
                ax.set_xlim(domain_min, domain_max)
                
                # Set y-axis range based on domain for better visualization
                if dataset == "V-A,V-M,V-NYT,V-T":
                    # For [-4,4] domain, use larger residual range
                    max_residual = max(abs(min(residuals)), abs(max(residuals)))
                    ax.set_ylim(-max_residual*1.1, max_residual*1.1)
                else:
                    # For [0,1] domain, use smaller residual range
                    max_residual = max(abs(min(residuals)), abs(max(residuals)))
                    ax.set_ylim(-max_residual*1.2, max_residual*1.2)
                
                # Add statistics
                mean_residual = np.mean(residuals)
                std_residual = np.std(residuals)
                ax.text(0.05, 0.95, f'Mean: {mean_residual:.3f}\nStd: {std_residual:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # Hide unused subplots
            for idx in range(len(dataset_data), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle(f'Residual Analysis for {dataset}', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            # Save plot
            safe_dataset_name = dataset.replace(',', '_').replace(' ', '_')
            plt.savefig(f"{save_path}/residuals_{safe_dataset_name}.png", dpi=300, bbox_inches='tight')
            plt.show()
            
            # Create domain-specific analysis for this dataset
            self.create_individual_domain_analysis(dataset, dataset_data, save_path)
            
            # Create summary statistics
            self.create_summary_stats(dataset, dataset_data, save_path)
    
    def create_summary_stats(self, dataset: str, dataset_data: Dict, save_path: str):
        """Create summary statistics for residuals."""
        stats_data = []
        
        for model, data in dataset_data.items():
            true_vals, pred_vals = zip(*data)
            residuals = self.calculate_residuals(true_vals, pred_vals)
            
            # Calculate metrics
            mae = np.mean(np.abs(residuals))
            mse = np.mean(np.square(residuals))
            rmse = np.sqrt(mse)
            mean_residual = np.mean(residuals)
            std_residual = np.std(residuals)
            
            # Calculate R²
            ss_res = np.sum(np.square(residuals))
            ss_tot = np.sum(np.square(np.array(true_vals) - np.mean(true_vals)))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            stats_data.append({
                'Dataset': dataset,
                'Model': model,
                'Samples': len(data),
                'MAE': mae,
                'MSE': mse,
                'RMSE': rmse,
                'Mean_Residual': mean_residual,
                'Std_Residual': std_residual,
                'R_squared': r_squared
            })
        
        # Save to CSV
        df = pd.DataFrame(stats_data)
        safe_dataset_name = dataset.replace(',', '_').replace(' ', '_')
        df.to_csv(f"{save_path}/stats_{safe_dataset_name}.csv", index=False)
        
        print(f"Summary statistics saved for {dataset}")
        print(df[['Model', 'Samples', 'MAE', 'RMSE', 'R_squared']].to_string(index=False))
    
    def create_individual_domain_analysis(self, dataset: str, dataset_data: Dict, save_path: str):
        """Create domain-specific analysis for individual datasets."""
        if not dataset_data:
            return
            
        # Prepare data for plotting
        plot_data = []
        for model, data in dataset_data.items():
            true_vals, pred_vals = zip(*data)
            residuals = self.calculate_residuals(true_vals, pred_vals)
            
            for true_val, pred_val, residual in zip(true_vals, pred_vals, residuals):
                plot_data.append({
                    'Model': model,
                    'True_Value': true_val,
                    'Prediction': pred_val,
                    'Residual': residual
                })
        
        df = pd.DataFrame(plot_data)
        domain_min, domain_max = self.dataset_domains[dataset]
        
        # Create comprehensive domain-specific plots
        fig, axes = styled_subplots(width=20.0, height=12.0, nrows=2, ncols=3)
        
        # 1. Residuals vs True Values (all models)
        for model in df['Model'].unique():
            subset = df[df['Model'] == model]
            axes[0,0].scatter(subset['True_Value'], subset['Residual'], 
                            alpha=0.6, label=model, s=15)
        axes[0,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,0].set_xlim(domain_min, domain_max)
        axes[0,0].set_xlabel('True Values')
        axes[0,0].set_ylabel('Residuals')
        axes[0,0].set_title(f'Residuals vs True Values\n{dataset}')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Predictions vs True Values (all models)
        for model in df['Model'].unique():
            subset = df[df['Model'] == model]
            axes[0,1].scatter(subset['True_Value'], subset['Prediction'], 
                            alpha=0.6, label=model, s=15)
        # Perfect prediction line
        axes[0,1].plot([domain_min, domain_max], [domain_min, domain_max], 
                      'r--', alpha=0.7, label='Perfect Prediction')
        axes[0,1].set_xlim(domain_min, domain_max)
        axes[0,1].set_ylim(domain_min, domain_max)
        axes[0,1].set_xlabel('True Values')
        axes[0,1].set_ylabel('Predictions')
        axes[0,1].set_title(f'Predictions vs True Values\n{dataset}')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Residual distribution by model
        sns.boxplot(data=df, x='Model', y='Residual', ax=axes[0,2])
        axes[0,2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0,2].set_title(f'Residual Distribution by Model\n{dataset}')
        axes[0,2].tick_params(axis='x', rotation=45)
        
        # 4. Residual histogram (all models combined)
        axes[1,0].hist(df['Residual'], bins=50, alpha=0.7, edgecolor='black')
        axes[1,0].axvline(x=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('Residuals')
        axes[1,0].set_ylabel('Frequency')
        axes[1,0].set_title(f'Residual Distribution\n{dataset}')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Model performance comparison (R²)
        model_stats = []
        for model in df['Model'].unique():
            subset = df[df['Model'] == model]
            true_vals = subset['True_Value'].values
            residuals = subset['Residual'].values
            
            # Calculate R²
            ss_res = np.sum(np.square(residuals))
            ss_tot = np.sum(np.square(true_vals - np.mean(true_vals)))
            r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
            
            model_stats.append({'Model': model, 'R_squared': r_squared})
        
        stats_df = pd.DataFrame(model_stats)
        bars = axes[1,1].bar(stats_df['Model'], stats_df['R_squared'])
        axes[1,1].set_ylabel('R² Score')
        axes[1,1].set_title(f'Model Performance (R²)\n{dataset}')
        axes[1,1].tick_params(axis='x', rotation=45)
        axes[1,1].grid(True, alpha=0.3)
        
        # Color bars based on performance
        for i, bar in enumerate(bars):
            if stats_df.iloc[i]['R_squared'] < 0:
                bar.set_color('red')
            elif stats_df.iloc[i]['R_squared'] < 0.5:
                bar.set_color('orange')
            else:
                bar.set_color('green')
        
        # 6. Residuals vs Predictions
        for model in df['Model'].unique():
            subset = df[df['Model'] == model]
            axes[1,2].scatter(subset['Prediction'], subset['Residual'], 
                            alpha=0.6, label=model, s=15)
        axes[1,2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,2].set_xlabel('Predictions')
        axes[1,2].set_ylabel('Residuals')
        axes[1,2].set_title(f'Residuals vs Predictions\n{dataset}')
        axes[1,2].legend()
        axes[1,2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Comprehensive Domain Analysis: {dataset} (Domain: [{domain_min}, {domain_max}])', 
                    fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save plot
        safe_dataset_name = dataset.replace(',', '_').replace(' ', '_')
        plt.savefig(f"{save_path}/domain_analysis_{safe_dataset_name}.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print model performance summary
        print(f"\nModel Performance Summary for {dataset}:")
        print("=" * 50)
        for _, row in stats_df.iterrows():
            status = "⚠️  POOR" if row['R_squared'] < 0 else "✅ GOOD" if row['R_squared'] > 0.5 else "⚠️  FAIR"
            print(f"{row['Model']:20} R² = {row['R_squared']:8.3f} {status}")
        print("=" * 50)
        
        # Add domain violation warning
        print(f"\nDomain Range: [{domain_min}, {domain_max}]")
        print("Note: Predictions outside domain have been clipped for analysis")
        print("Check domain_check_results.csv for detailed violation statistics")
    
    def plot_combined_residuals(self, save_path: str = "residual_plots"):
        """Create combined residual plots across all datasets."""
        os.makedirs(save_path, exist_ok=True)
        
        all_data = []
        
        # Collect all data
        for dataset in self.datasets:
            dataset_data = self.load_dataset_data(dataset)
            for model, data in dataset_data.items():
                true_vals, pred_vals = zip(*data)
                residuals = self.calculate_residuals(true_vals, pred_vals)
                
                for true_val, pred_val, residual in zip(true_vals, pred_vals, residuals):
                    all_data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'True_Value': true_val,
                        'Prediction': pred_val,
                        'Residual': residual
                    })
        
        if not all_data:
            print("No data found for combined analysis")
            return
            
        df = pd.DataFrame(all_data)
        
        # Create combined plots
        fig, axes = styled_subplots(width=16.0, height=12.0, nrows=2, ncols=2)
        
        # 1. Residuals by dataset
        sns.boxplot(data=df, x='Dataset', y='Residual', ax=axes[0,0])
        axes[0,0].set_title('Residual Distribution by Dataset')
        axes[0,0].tick_params(axis='x', rotation=45)
        
        # 2. Residuals by model
        sns.boxplot(data=df, x='Model', y='Residual', ax=axes[0,1])
        axes[0,1].set_title('Residual Distribution by Model')
        axes[0,1].tick_params(axis='x', rotation=45)
        
        # 3. Residuals vs True Values (colored by dataset)
        for dataset in df['Dataset'].unique():
            subset = df[df['Dataset'] == dataset]
            axes[1,0].scatter(subset['True_Value'], subset['Residual'], 
                            alpha=0.6, label=dataset, s=10)
        axes[1,0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1,0].set_xlabel('True Values')
        axes[1,0].set_ylabel('Residuals')
        axes[1,0].set_title('Residuals vs True Values (by Dataset)')
        axes[1,0].legend()
        axes[1,0].grid(True, alpha=0.3)
        
        # Set x-axis limits to show the full range of all datasets
        all_true_values = df['True_Value'].values
        axes[1,0].set_xlim(min(all_true_values), max(all_true_values))
        
        # 4. Heatmap of residuals by dataset and model
        pivot_data = df.groupby(['Dataset', 'Model'])['Residual'].agg(['mean', 'std']).reset_index()
        pivot_mean = pivot_data.pivot(index='Model', columns='Dataset', values='mean')
        sns.heatmap(pivot_mean, annot=True, fmt='.3f', cmap='RdBu_r', center=0, ax=axes[1,1])
        axes[1,1].set_title('Mean Residuals by Dataset and Model')
        
        plt.suptitle('Combined Residual Analysis Across All Datasets', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{save_path}/combined_residuals.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        # Save combined data
        df.to_csv(f"{save_path}/combined_residual_data.csv", index=False)
        print(f"Combined analysis saved to {save_path}")
        
        # Create domain-specific comparison plots
        self.create_domain_comparison_plots(df, save_path)
    
    def create_domain_comparison_plots(self, df: pd.DataFrame, save_path: str):
        """Create domain-specific comparison plots for better visualization."""
        # Separate datasets by domain
        zero_one_datasets = df[df['Dataset'].isin(['EI-reg', 'V-reg', 'SST'])]
        neg_four_four_datasets = df[df['Dataset'] == 'V-A,V-M,V-NYT,V-T']
        
        # Plot for [0,1] domain datasets
        if not zero_one_datasets.empty:
            fig, axes = styled_subplots(width=16.0, height=6.0, nrows=1, ncols=2)
            
            # Residuals vs True Values for [0,1] domain
            for dataset in zero_one_datasets['Dataset'].unique():
                subset = zero_one_datasets[zero_one_datasets['Dataset'] == dataset]
                axes[0].scatter(subset['True_Value'], subset['Residual'], 
                              alpha=0.6, label=dataset, s=15)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0].set_xlim(0, 1)
            axes[0].set_xlabel('True Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs True Values\n[0,1] Domain Datasets')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Box plot for [0,1] domain
            sns.boxplot(data=zero_one_datasets, x='Dataset', y='Residual', ax=axes[1])
            axes[1].set_title('Residual Distribution\n[0,1] Domain Datasets')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Domain-Specific Analysis: [0,1] Range', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{save_path}/domain_0_1_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()
        
        # Plot for [-4,4] domain dataset
        if not neg_four_four_datasets.empty:
            fig, axes = styled_subplots(width=16.0, height=6.0, nrows=1, ncols=2)
            
            # Residuals vs True Values for [-4,4] domain
            for model in neg_four_four_datasets['Model'].unique():
                subset = neg_four_four_datasets[neg_four_four_datasets['Model'] == model]
                axes[0].scatter(subset['True_Value'], subset['Residual'], 
                              alpha=0.6, label=model, s=15)
            axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
            axes[0].set_xlim(-4, 4)
            axes[0].set_xlabel('True Values')
            axes[0].set_ylabel('Residuals')
            axes[0].set_title('Residuals vs True Values\n[-4,4] Domain Dataset')
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Box plot for [-4,4] domain
            sns.boxplot(data=neg_four_four_datasets, x='Model', y='Residual', ax=axes[1])
            axes[1].set_title('Residual Distribution by Model\n[-4,4] Domain Dataset')
            axes[1].tick_params(axis='x', rotation=45)
            
            plt.suptitle('Domain-Specific Analysis: [-4,4] Range', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(f"{save_path}/domain_-4_4_comparison.png", dpi=300, bbox_inches='tight')
            plt.show()

def main():
    """Main function to run the residual analysis."""
    analyzer = ResidualAnalyzer()
    
    print("Starting residual analysis...")
    print("=" * 50)
    
    # Create individual dataset plots
    analyzer.plot_residuals_by_dataset()
    
    print("\n" + "=" * 50)
    print("Creating combined analysis...")
    
    # Create combined plots
    analyzer.plot_combined_residuals()
    
    print("\nAnalysis complete! Check the 'residual_plots' directory for results.")

if __name__ == "__main__":
    main()

