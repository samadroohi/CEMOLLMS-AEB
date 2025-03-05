import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from pathlib import Path
import sys

# Add the parent directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)  # src directory
sys.path.append(parent_dir)

# Now import config from the src directory
from config import Config

def load_model_metrics(dataset, models, temp="0.9"):
    """Load metrics for all models for a specific dataset."""
    all_metrics = {}
    
    for model in models:
        model_name = model.split('/')[-1]
        metrics_path = f"results/plots/{dataset}/temp_{temp}/{model_name}/combined_metrics.json"
        
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
                all_metrics[model_name] = metrics
        else:
            print(f"Warning: No metrics found for {model_name} on {dataset}")
    
    return all_metrics

def plot_psize_vs_confidence(all_metrics, dataset, output_dir="results/integrated_analysis"):
    """Plot prediction set size vs confidence (1-alpha) for all models."""
    try:
        from adjustText import adjust_text  # For text annotation positioning
    except ImportError:
        print("Please install adjustText with: pip install adjustText")
        # Fallback function without adjust_text
        def adjust_text(texts):
            pass
    
    fig, ax = plt.subplots(figsize=(12, 7))  # Increased figure width for legend
    all_texts = []
    
    for model_name, metrics in all_metrics.items():
        # Determine if dataset is regression or classification
        is_regression = any(key in metrics for key in ['mse', 'rmse', 'mae'])
        
        # Get confidence values (handle both alpha and confidence formats)
        if 'alpha' in metrics:
            confidence = [1-a for a in metrics['alpha']]
        elif 'confidences' in metrics:
            confidence = metrics['confidences']
        else:
            print(f"Warning: No confidence/alpha values for {model_name} on {dataset}")
            continue
        
        # Get prediction set sizes (handle both psize and interval_sizes formats)
        if 'psize' in metrics:
            psize = metrics['psize']
        elif 'average_interval_sizes' in metrics:
            psize = metrics['average_interval_sizes']
        else:
            print(f"Warning: No prediction set sizes for {model_name} on {dataset}")
            continue
        
        # Get coverage (different name in regression vs classification)
        if 'coverage' in metrics:
            coverage = metrics['coverage']
        elif 'coverages' in metrics:
            coverage = metrics['coverages']
        else:
            print(f"Warning: No coverage values for {model_name} on {dataset}")
            continue
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'confidence': confidence,
            'psize': psize,
            'coverage': coverage
        })
        
        # Plot points with smaller fixed size and lines
        ax.scatter(df['confidence'], df['psize'], s=30, alpha=0.7)  # Smaller fixed size points
        ax.plot(df['confidence'], df['psize'], '-', alpha=0.7, label=model_name)
        
        # Add text annotations for coverage
        for i, row in df.iterrows():
            # Only annotate some points to avoid overcrowding 
            if i % 2 == 0:  # Annotate every other point
                txt = ax.text(
                    row['confidence'], 
                    row['psize'], 
                    f"{row['coverage']:.2f}", 
                    fontsize=8,
                    ha='center', 
                    va='bottom'
                )
                all_texts.append(txt)
    
    # Adjust text positions to avoid overlaps
    if all_texts:
        adjust_text(all_texts, arrowprops=dict(arrowstyle='->', color='red'))
    
    ax.set_title(f'Prediction Set Size vs Confidence for {dataset}')
    ax.set_xlabel('Confidence (1-α)')
    ax.set_ylabel('Prediction Set Size')
    ax.grid(True, alpha=0.3)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create a specific folder for psize vs confidence plots
    psize_dir = os.path.join(output_dir, "psize_vs_confidence")
    os.makedirs(psize_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(f"{psize_dir}/{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_confidence_vs_coverage(all_metrics, dataset, output_dir="results/integrated_analysis"):
    """Plot confidence (1-alpha) vs empirical coverage for all models."""
    fig, ax = plt.subplots(figsize=(12, 7))  # Increased figure width for legend
    
    # Add diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    for model_name, metrics in all_metrics.items():
        # Determine if dataset is regression or classification
        is_regression = any(key in metrics for key in ['mse', 'rmse', 'mae'])
        
        # Get confidence values (handle both alpha and confidence formats)
        if 'alpha' in metrics:
            confidence = [1-a for a in metrics['alpha']]
        elif 'confidences' in metrics:
            confidence = metrics['confidences']
        else:
            print(f"Warning: No confidence/alpha values for {model_name} on {dataset}")
            continue
        
        # Get coverage (different name in regression vs classification)
        if 'coverage' in metrics:
            coverage = metrics['coverage']
        elif 'coverages' in metrics:
            coverage = metrics['coverages']
        else:
            print(f"Warning: No coverage values for {model_name} on {dataset}")
            continue
        
        # Create DataFrame for plotting
        df = pd.DataFrame({
            'confidence': confidence,
            'coverage': coverage
        })
        
        # Sort by confidence for proper line plotting
        df = df.sort_values('confidence')
        
        # Plot confidence vs coverage
        ax.scatter(df['confidence'], df['coverage'], s=30, alpha=0.7)
        ax.plot(df['confidence'], df['coverage'], '-', alpha=0.7, label=model_name)
    
    ax.set_title(f'Confidence vs Empirical Coverage for {dataset}')
    ax.set_xlabel('Confidence (1-α)')
    ax.set_ylabel('Empirical Coverage')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create a specific folder for confidence vs coverage plots
    coverage_dir = os.path.join(output_dir, "confidence_vs_coverage")
    os.makedirs(coverage_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(f"{coverage_dir}/{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close()

def generate_performance_tables(all_metrics_by_dataset, output_dir="results/integrated_analysis"):
    """Generate summary tables for different task types."""
    # Create task groups
    regression_datasets = ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank", "SST"]
    ordinal_datasets = ["EI-oc", "TDT", "V-oc", "SST5"]
    multiclass_datasets = ["GoEmotions", "E-c"]
    
    # Create output directory for tables
    tables_dir = os.path.join(output_dir, "tables")
    os.makedirs(tables_dir, exist_ok=True)
    
    # Generate regression table
    generate_regression_table(all_metrics_by_dataset, regression_datasets, tables_dir)
    
    # Generate ordinal classification table
    generate_ordinal_table(all_metrics_by_dataset, ordinal_datasets, tables_dir)
    
    # Generate multiclass classification table
    generate_multiclass_table(all_metrics_by_dataset, multiclass_datasets, tables_dir)
    
    # Generate calibration metrics table
    generate_calibration_table(all_metrics_by_dataset, tables_dir)
    
    # Generate conformal prediction metrics table
    generate_conformal_metrics_table(all_metrics_by_dataset, tables_dir)

def generate_regression_table(all_metrics_by_dataset, datasets, output_dir):
    """Generate a summary table for regression tasks."""
    # Initialize a multi-index DataFrame
    all_models = set()
    for dataset in datasets:
        if dataset in all_metrics_by_dataset:
            all_models.update(all_metrics_by_dataset[dataset].keys())
    
    all_models = sorted(list(all_models))
    
    # Create index for rows (models)
    index = pd.Index(all_models, name="Model")
    
    # Create MultiIndex for columns (dataset, metric)
    metrics = ["pearson_correlation", "mse"]
    column_tuples = [(dataset, metric) for dataset in datasets for metric in metrics]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dataset", "Metric"])
    
    # Create DataFrame
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame with metrics
    for dataset in datasets:
        if dataset not in all_metrics_by_dataset:
            continue
            
        for model in all_models:
            if model not in all_metrics_by_dataset[dataset]:
                continue
                
            metrics_data = all_metrics_by_dataset[dataset][model]
            
            # Extract metrics (handle different metric names)
            if "pearson_correlation" in metrics_data:
                df.loc[model, (dataset, "pearson_correlation")] = metrics_data["pearson_correlation"]
            elif "pearson" in metrics_data:
                df.loc[model, (dataset, "pearson_correlation")] = metrics_data["pearson"]
            
            if "mse" in metrics_data:
                df.loc[model, (dataset, "mse")] = metrics_data["mse"]
    
    # Save table
    df.to_csv(os.path.join(output_dir, "regression_performance.csv"))
    print(f"Saved regression performance table to {output_dir}/regression_performance.csv")
    
    return df

def generate_ordinal_table(all_metrics_by_dataset, datasets, output_dir):
    """Generate a summary table for ordinal classification tasks."""
    # Initialize a multi-index DataFrame
    all_models = set()
    for dataset in datasets:
        if dataset in all_metrics_by_dataset:
            all_models.update(all_metrics_by_dataset[dataset].keys())
    
    all_models = sorted(list(all_models))
    
    # Create index for rows (models)
    index = pd.Index(all_models, name="Model")
    
    # Create MultiIndex for columns (dataset, metric)
    metrics = ["accuracy", "macro_f1"]
    column_tuples = [(dataset, metric) for dataset in datasets for metric in metrics]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dataset", "Metric"])
    
    # Create DataFrame
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame with metrics
    for dataset in datasets:
        if dataset not in all_metrics_by_dataset:
            continue
            
        for model in all_models:
            if model not in all_metrics_by_dataset[dataset]:
                continue
                
            metrics_data = all_metrics_by_dataset[dataset][model]
            
            # Extract metrics
            if "accuracy" in metrics_data:
                df.loc[model, (dataset, "accuracy")] = metrics_data["accuracy"]
            
            if "macro_f1" in metrics_data:
                df.loc[model, (dataset, "macro_f1")] = metrics_data["macro_f1"]
            elif "f1_macro" in metrics_data:
                df.loc[model, (dataset, "macro_f1")] = metrics_data["f1_macro"]
    
    # Save table
    df.to_csv(os.path.join(output_dir, "ordinal_performance.csv"))
    print(f"Saved ordinal classification performance table to {output_dir}/ordinal_performance.csv")
    
    return df

def generate_multiclass_table(all_metrics_by_dataset, datasets, output_dir):
    """Generate a summary table for multiclass classification tasks."""
    # Initialize a multi-index DataFrame
    all_models = set()
    for dataset in datasets:
        if dataset in all_metrics_by_dataset:
            all_models.update(all_metrics_by_dataset[dataset].keys())
    
    all_models = sorted(list(all_models))
    
    # Create index for rows (models)
    index = pd.Index(all_models, name="Model")
    
    # Create MultiIndex for columns (dataset, metric)
    metrics = ["jaccard_similarity", "f1_micro", "f1_macro"]
    column_tuples = [(dataset, metric) for dataset in datasets for metric in metrics]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dataset", "Metric"])
    
    # Create DataFrame
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame with metrics
    for dataset in datasets:
        if dataset not in all_metrics_by_dataset:
            continue
            
        for model in all_models:
            if model not in all_metrics_by_dataset[dataset]:
                continue
                
            metrics_data = all_metrics_by_dataset[dataset][model]
            
            # Extract metrics
            if "jaccard_similarity" in metrics_data:
                df.loc[model, (dataset, "jaccard_similarity")] = metrics_data["jaccard_similarity"]
            elif "jaccard_index" in metrics_data:
                df.loc[model, (dataset, "jaccard_similarity")] = metrics_data["jaccard_index"]
            
            if "f1_micro" in metrics_data:
                df.loc[model, (dataset, "f1_micro")] = metrics_data["f1_micro"]
            
            if "f1_macro" in metrics_data:
                df.loc[model, (dataset, "f1_macro")] = metrics_data["f1_macro"]
    
    # Save table
    df.to_csv(os.path.join(output_dir, "multiclass_performance.csv"))
    print(f"Saved multiclass classification performance table to {output_dir}/multiclass_performance.csv")
    
    return df

def generate_calibration_table(all_metrics_by_dataset, output_dir):
    """Generate a summary table for calibration metrics (ECE, MCALE)."""
    # Get all datasets and models
    all_datasets = list(all_metrics_by_dataset.keys())
    all_models = set()
    for dataset in all_datasets:
        all_models.update(all_metrics_by_dataset[dataset].keys())
    
    all_models = sorted(list(all_models))
    all_datasets = sorted(all_datasets)
    
    # Create index for rows (models)
    index = pd.Index(all_models, name="Model")
    
    # Create MultiIndex for columns (dataset, metric)
    metrics = ["ece", "mcale"]
    column_tuples = [(dataset, metric) for dataset in all_datasets for metric in metrics]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dataset", "Metric"])
    
    # Create DataFrame
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame with metrics
    for dataset in all_datasets:
        for model in all_models:
            if model not in all_metrics_by_dataset[dataset]:
                continue
                
            metrics_data = all_metrics_by_dataset[dataset][model]
            
            # Extract metrics (handle different metric names)
            if "ece" in metrics_data:
                df.loc[model, (dataset, "ece")] = metrics_data["ece"]
                
            if "mcale" in metrics_data:
                df.loc[model, (dataset, "mcale")] = metrics_data["mcale"]
    
    # Save table
    df.to_csv(os.path.join(output_dir, "calibration_metrics.csv"))
    print(f"Saved calibration metrics table to {output_dir}/calibration_metrics.csv")
    
    return df

def generate_conformal_metrics_table(all_metrics_by_dataset, output_dir):
    """Generate a summary table for conformal prediction metrics (mean confidence, ACE, MCOVE)."""
    # Get all datasets and models
    all_datasets = list(all_metrics_by_dataset.keys())
    all_models = set()
    for dataset in all_datasets:
        all_models.update(all_metrics_by_dataset[dataset].keys())
    
    all_models = sorted(list(all_models))
    all_datasets = sorted(all_datasets)
    
    # Create index for rows (models)
    index = pd.Index(all_models, name="Model")
    
    # Create MultiIndex for columns (dataset, metric)
    metrics = ["mean_confidence", "ace", "mcove"]
    column_tuples = [(dataset, metric) for dataset in all_datasets for metric in metrics]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dataset", "Metric"])
    
    # Create DataFrame
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame with metrics
    for dataset in all_datasets:
        for model in all_models:
            if model not in all_metrics_by_dataset[dataset]:
                continue
                
            metrics_data = all_metrics_by_dataset[dataset][model]
            
            # Extract metrics
            if "mean_confidence" in metrics_data:
                df.loc[model, (dataset, "mean_confidence")] = metrics_data["mean_confidence"]
            
            if "ace" in metrics_data:
                df.loc[model, (dataset, "ace")] = metrics_data["ace"]
                
            if "mcove" in metrics_data:
                df.loc[model, (dataset, "mcove")] = metrics_data["mcove"]
    
    # Save table
    df.to_csv(os.path.join(output_dir, "conformal_metrics.csv"))
    print(f"Saved conformal prediction metrics table to {output_dir}/conformal_metrics.csv")
    
    return df

def run_integrated_analysis():
    # Models to compare
    models = [
        "lzw1008/Emollama-7b",
        "lzw1008/Emollama-chat-7b",
        "lzw1008/Emollama-chat-13b",
        "lzw1008/Emoopt-13b",
        "lzw1008/Emobloom-7b",
    ]
    
    # Datasets to analyze
    datasets = [
        "EI-oc", 
        "TDT", 
        "SST5",
        "V-oc",  
        "EI-reg", 
        "V-reg", 
        "V-A,V-M,V-NYT,V-T", 
        "Emobank", 
        "SST", 
        "GoEmotions", 
        "E-c"
    ]
    
    output_dir = "results/integrated_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # Store all metrics by dataset
    all_metrics_by_dataset = {}
    
    # For each dataset, generate the comparison plots
    for dataset in datasets:
        print(f"Processing {dataset}...")
        all_metrics = load_model_metrics(dataset, models)
        
        if all_metrics:
            # Store metrics for later table generation
            all_metrics_by_dataset[dataset] = all_metrics
            
            # Plot prediction set size vs confidence
            plot_psize_vs_confidence(all_metrics, dataset, output_dir)
            
            # Plot confidence vs empirical coverage
            plot_confidence_vs_coverage(all_metrics, dataset, output_dir)
            
            print(f"Generated plots for {dataset}")
        else:
            print(f"No metrics found for {dataset}")
    
    # Generate performance tables
    generate_performance_tables(all_metrics_by_dataset, output_dir)

if __name__ == "__main__":
    run_integrated_analysis()
