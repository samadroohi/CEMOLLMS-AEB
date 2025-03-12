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
    metrics = ["accuracy", "macro_f1", "pearson_correlation"]
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
            
            # Add Pearson correlation for ordinal tasks
            if "pearson_correlation" in metrics_data:
                df.loc[model, (dataset, "pearson_correlation")] = metrics_data["pearson_correlation"]
            elif "pearson" in metrics_data:
                df.loc[model, (dataset, "pearson_correlation")] = metrics_data["pearson"]
    
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
    """Generate a summary table for conformal prediction metrics (mean confidence, ACE, MCovE, avg_size)."""
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
    metrics = ["mean_confidence", "ace", "mcove", "avg_size"]
    column_tuples = [(dataset, metric) for dataset in all_datasets for metric in metrics]
    columns = pd.MultiIndex.from_tuples(column_tuples, names=["Dataset", "Metric"])
    
    # Create DataFrame
    df = pd.DataFrame(index=index, columns=columns)
    
    # Fill DataFrame with metrics
    for dataset in all_datasets:
        if dataset not in all_metrics_by_dataset:
            continue
            
        for model in all_models:
            if model not in all_metrics_by_dataset[dataset]:
                continue
                
            metrics_data = all_metrics_by_dataset[dataset][model]
            model_name = model
            
            # Calculate ACE and MCovE if not already present
            if 'ace' not in metrics_data or 'mcove' not in metrics_data:
                # Get confidence values and coverage
                confidences = []
                coverages = []
                
                if 'alpha' in metrics_data:
                    confidences = [1 - alpha for alpha in metrics_data['alpha']]
                elif 'confidences' in metrics_data:
                    confidences = metrics_data['confidences']
                
                if 'coverage' in metrics_data:
                    coverages = metrics_data['coverage']
                elif 'coverages' in metrics_data:
                    coverages = metrics_data['coverages']
                
                # Calculate ACE and MCovE if we have both confidences and coverages
                if confidences and coverages:
                    abs_errors = [abs(cov - conf) for conf, cov in zip(confidences, coverages)]
                    ace = sum(abs_errors) / len(abs_errors)
                    mcove = max(abs_errors)
                    
                    # Store calculated metrics
                    metrics_data['ace'] = ace
                    metrics_data['mcove'] = mcove
            
            # Store metrics in the DataFrame
            if 'ace' in metrics_data:
                df.loc[model, (dataset, "ace")] = metrics_data["ace"]
            
            if 'mcove' in metrics_data:
                df.loc[model, (dataset, "mcove")] = metrics_data["mcove"]
            
            # Calculate and store mean confidence
            confidences = []
            if 'alpha' in metrics_data:
                confidences = [1 - alpha for alpha in metrics_data['alpha']]
            elif 'confidences' in metrics_data:
                confidences = metrics_data['confidences']
                
            if confidences:
                mean_confidence = sum(confidences) / len(confidences)
                df.loc[model, (dataset, "mean_confidence")] = mean_confidence
            
            # Try to get size data from conformal prediction results
            conformal_path = f"results/conformal_results/{dataset}/temp_0.9/{model_name}.json"
            
            avg_size = None
            
            # First try to get size from the metrics_data
            if 'psize' in metrics_data:
                sizes = metrics_data['psize']
                avg_size = sum(sizes) / len(sizes)
            elif 'average_interval_sizes' in metrics_data:
                sizes = metrics_data['average_interval_sizes']
                avg_size = sum(sizes) / len(sizes)
            
            # If not found, try to load from conformal results file
            if avg_size is None and os.path.exists(conformal_path):
                try:
                    with open(conformal_path, 'r') as f:
                        conformal_data = json.load(f)
                    
                    # Extract sizes from conformal prediction results
                    # Different alphas have different sizes
                    all_sizes = []
                    
                    for alpha_key in conformal_data:
                        if alpha_key.replace('.', '', 1).isdigit():  # Check if it's an alpha key
                            alpha_data = conformal_data[alpha_key]
                            
                            # Check for prediction_sets (classification) or intervals (regression)
                            if 'prediction_sets' in alpha_data:
                                sizes = [len(pred_set) for pred_set in alpha_data['prediction_sets']]
                                if sizes:
                                    all_sizes.append(np.mean(sizes))
                            elif 'intervals' in alpha_data:
                                sizes = [interval[1] - interval[0] for interval in alpha_data['intervals']]
                                if sizes:
                                    all_sizes.append(np.mean(sizes))
                    
                    if all_sizes:
                        avg_size = sum(all_sizes) / len(all_sizes)
                except Exception as e:
                    print(f"Error loading conformal data for {model_name} on {dataset}: {e}")
            
            if avg_size is not None:
                df.loc[model, (dataset, "avg_size")] = avg_size
    
    # Save table
    df.to_csv(os.path.join(output_dir, "conformal_metrics.csv"))
    print(f"Saved conformal metrics table to {output_dir}/conformal_metrics.csv")
    
    return df

def load_reliability_data(dataset, models, temp="0.9"):
    """Load conformal prediction results for all models for a specific dataset."""
    all_reliability_data = {}
    
    for model in models:
        model_name = model.split('/')[-1]
        reliability_path = f"results/conformal_results/{dataset}/temp_{temp}/{model_name}.json"
        
        if os.path.exists(reliability_path):
            with open(reliability_path, 'r') as f:
                reliability_data = json.load(f)
                all_reliability_data[model_name] = reliability_data
        else:
            print(f"Warning: No reliability data found for {model_name} on {dataset}")
    
    return all_reliability_data

def plot_merged_reliability_diagram(all_reliability_data, dataset, output_dir="results/integrated_analysis"):
    """Plot merged reliability diagram for all models for a specific dataset."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Add diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    for model_name, reliability_data in all_reliability_data.items():
        print(f"Processing {model_name} for {dataset}...")
        
        # Check if there are probabilities in the data
        has_probs = False
        probs = []
        true_values = []
        
        # First, check if there's a single entry at the top level with probabilities
        if 'probs' in reliability_data:
            probs = reliability_data['probs']
            true_values = reliability_data.get('true_values', [])
            has_probs = True
        else:
            # Look in alpha keys for probabilities
            alpha_keys = [k for k in reliability_data.keys() if k.replace(".", "").isdigit()]
            
            if alpha_keys and 'probs' in reliability_data[alpha_keys[0]]:
                # Use probabilities from the first alpha value
                alpha_key = alpha_keys[0]
                probs = reliability_data[alpha_key]['probs']
                true_values = reliability_data[alpha_key].get('true_values', [])
                has_probs = True
        
        if not has_probs or not probs:
            print(f"Warning: No probability data found for {model_name} on {dataset}")
            continue
        
        # Process probabilities to create reliability diagram data
        confidences = []
        correctness = []
        
        # Debug info
        print(f"Number of examples with probs: {len(probs)}")
        if probs and len(probs) > 0:
            print(f"Type of first prob: {type(probs[0])}")
            if isinstance(probs[0], list) and len(probs[0]) > 0:
                print(f"Type of first element in first prob: {type(probs[0][0])}")
        
        # For classification tasks, each "prob" is a distribution over classes
        for i, prob_dist in enumerate(probs):
            # Skip empty distributions
            if not prob_dist:
                continue
                
            # Check the structure of prob_dist
            if isinstance(prob_dist, list) and len(prob_dist) > 0:
                # Check if it's nested (multi-label case)
                if isinstance(prob_dist[0], list):
                    # Handle multi-label case: multiple distributions per example
                    for j, class_probs in enumerate(prob_dist):
                        if class_probs and len(class_probs) > 0:
                            # Get confidence as max probability
                            confidence = max(class_probs)
                            pred_class = class_probs.index(max(class_probs))
                            
                            # Check if prediction was correct
                            # This is a placeholder - adjust based on your data format
                            correct = 0
                            if i < len(true_values) and isinstance(true_values[i], list):
                                # Assuming true_values is a list of lists for multi-label
                                if pred_class in true_values[i]:
                                    correct = 1
                            
                            confidences.append(confidence)
                            correctness.append(correct)
                else:
                    # Single distribution per example
                    confidence = max(prob_dist)
                    pred_class = prob_dist.index(max(prob_dist))
                    
                    # Check if prediction was correct
                    correct = 0
                    if i < len(true_values):
                        if isinstance(true_values[i], list):
                            # Multi-label case where true values are lists of labels
                            if pred_class in true_values[i]:
                                correct = 1
                        else:
                            # Single-label case
                            if pred_class == true_values[i]:
                                correct = 1
                    
                    confidences.append(confidence)
                    correctness.append(correct)
            else:
                print(f"Warning: Unexpected probability format for example {i}")
        
        print(f"Extracted {len(confidences)} confidence values for {model_name}")
        
        if not confidences:
            print(f"Warning: Could not extract confidence values for {model_name} on {dataset}")
            continue
        
        # Create bins
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(confidences, bin_edges) - 1
        
        # Calculate accuracy per bin
        bin_confidences = []
        bin_accuracies = []
        
        for bin_idx in range(num_bins):
            bin_mask = (bin_indices == bin_idx)
            if np.sum(bin_mask) > 0:  # Skip empty bins
                bin_conf = np.mean(np.array(confidences)[bin_mask])
                bin_acc = np.mean(np.array(correctness)[bin_mask])
                
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
        
        print(f"Created {len(bin_confidences)} bins with data for {model_name}")
        
        # Create DataFrame and sort by confidence
        df = pd.DataFrame({
            'bin_confidence': bin_confidences,
            'bin_accuracy': bin_accuracies
        })
        df = df.sort_values('bin_confidence')
        
        # Plot reliability diagram
        ax.scatter(df['bin_confidence'], df['bin_accuracy'], s=30, alpha=0.7)
        ax.plot(df['bin_confidence'], df['bin_accuracy'], '-', alpha=0.7, label=model_name)
    
    ax.set_title(f'Reliability Diagram for {dataset}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create a specific folder for reliability diagrams
    reliability_dir = os.path.join(output_dir, "reliability_diagrams")
    os.makedirs(reliability_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(f"{reliability_dir}/{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close()

def plot_multilabel_reliability_diagram(all_reliability_data, dataset, output_dir="results/integrated_analysis"):
    """Plot reliability diagram specifically for multi-label datasets like E-c and GoEmotions."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    # Add diagonal reference line (perfect calibration)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Perfect calibration')
    
    for model_name, reliability_data in all_reliability_data.items():
        print(f"\nProcessing multi-label data for {model_name} on {dataset}...")
        
        # Only use alpha 0.1 since probs are the same across alpha values
        if '0.1' not in reliability_data:
            print(f"Warning: Alpha 0.1 not found for {model_name} on {dataset}")
            continue
            
        data = reliability_data['0.1']
        
        if 'probs' not in data or 'true_values' not in data:
            print(f"Warning: Missing required fields for {model_name} on {dataset}")
            continue
            
        # Extract data
        probs = data['probs']
        true_values = data['true_values']
        predictions = data.get('predictions', [])
        
        print(f"Found {len(probs)} examples with probability data")
        print(f"Found {len(true_values)} examples with true values")
        
        # Print detailed information about a few examples to understand structure
        if len(probs) > 0 and len(true_values) > 0:
            print(f"\nExample 0:")
            print(f"  True values: {true_values[0]}")
            if len(probs[0]) > 0:
                top_probs = sorted([(i, p) for i, p in enumerate(probs[0][0])], key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top 3 probs: {top_probs}")
            if len(predictions) > 0:
                print(f"  Prediction: {predictions[0]}")
            
            # Print class mapping for this dataset
            if dataset in Config.VALID_D_TYPES:
                print(f"\nClass mapping for {dataset}:")
                for idx, label in Config.VALID_D_TYPES[dataset].items():
                    print(f"  {idx}: {label}")
        
        # Process multi-label data
        all_confidences = []
        all_correctness = []
        
        # For multi-label classification, process each example
        for i, example_probs in enumerate(probs):
            if i >= len(true_values):
                continue
                
            true_labels = true_values[i]
            
            # Check if true_labels is a list of indices or strings
            true_label_indices = []
            for label in true_labels:
                if isinstance(label, str) and label.isdigit():
                    true_label_indices.append(int(label))
                elif isinstance(label, int):
                    true_label_indices.append(label)
                # If it's a string label, try to find its index
                elif isinstance(label, str) and dataset in Config.VALID_D_TYPES:
                    # Find index by value
                    for idx, val in Config.VALID_D_TYPES[dataset].items():
                        if val == label or val.endswith(label):
                            if idx.isdigit():
                                true_label_indices.append(int(idx))
            
            # Handle nested probability arrays (multiple labels per example)
            if isinstance(example_probs, list) and len(example_probs) > 0:
                for label_idx, class_probs in enumerate(example_probs):
                    if not class_probs or len(class_probs) == 0:
                        continue
                        
                    # Get confidence as max probability
                    confidence = max(class_probs)
                    pred_class_idx = class_probs.index(max(class_probs))
                    
                    # Check if prediction is in true labels
                    is_correct = 0
                    if pred_class_idx in true_label_indices:
                        is_correct = 1
                    
                    all_confidences.append(confidence)
                    all_correctness.append(is_correct)
        
        print(f"Extracted {len(all_confidences)} confidence values")
        print(f"Average correctness: {np.mean(all_correctness) if all_correctness else 'N/A'}")
        
        if not all_confidences:
            print(f"Warning: Could not extract confidence values for {model_name}")
            continue
        
        # Create bins
        num_bins = 10
        bin_edges = np.linspace(0, 1, num_bins + 1)
        bin_indices = np.digitize(all_confidences, bin_edges) - 1
        
        # Calculate accuracy per bin
        bin_confidences = []
        bin_accuracies = []
        bin_counts = []
        
        for bin_idx in range(num_bins):
            bin_mask = (bin_indices == bin_idx)
            if np.sum(bin_mask) > 0:  # Skip empty bins
                bin_conf = np.mean(np.array(all_confidences)[bin_mask])
                bin_acc = np.mean(np.array(all_correctness)[bin_mask])
                bin_count = np.sum(bin_mask)
                
                bin_confidences.append(bin_conf)
                bin_accuracies.append(bin_acc)
                bin_counts.append(bin_count)
                
                print(f"Bin {bin_idx}: conf={bin_conf:.2f}, acc={bin_acc:.2f}, count={bin_count}")
        
        print(f"Created {len(bin_confidences)} bins with data")
        
        # Create DataFrame and sort by confidence
        df = pd.DataFrame({
            'bin_confidence': bin_confidences,
            'bin_accuracy': bin_accuracies
        })
        df = df.sort_values('bin_confidence')
        
        # Plot reliability diagram
        ax.scatter(df['bin_confidence'], df['bin_accuracy'], s=30, alpha=0.7)
        ax.plot(df['bin_confidence'], df['bin_accuracy'], '-', alpha=0.7, label=model_name)
    
    ax.set_title(f'Reliability Diagram for {dataset}')
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3)
    
    # Move legend outside the plot
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0.)
    
    # Adjust layout to make room for the legend
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    
    # Create a specific folder for reliability diagrams
    reliability_dir = os.path.join(output_dir, "reliability_diagrams")
    os.makedirs(reliability_dir, exist_ok=True)
    
    # Save the plot
    plt.savefig(f"{reliability_dir}/{dataset}.png", dpi=300, bbox_inches='tight')
    plt.close()

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
    
    # Ordinal and regular classification datasets
    ordinal_datasets = [
        "EI-oc", 
        "TDT", 
        "SST5",
        "V-oc",
    ]
    
    # Multi-label classification datasets
    multilabel_datasets = [
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
    
    # Generate reliability diagrams for ordinal classification datasets
    for dataset in ordinal_datasets:
        print(f"Processing reliability diagram for {dataset}...")
        reliability_data = load_reliability_data(dataset, models)
        
        if reliability_data:
            plot_merged_reliability_diagram(reliability_data, dataset, output_dir)
            print(f"Generated reliability diagram for {dataset}")
        else:
            print(f"No reliability data found for {dataset}")
    
    # Generate multi-label reliability diagrams for E-c and GoEmotions
    for dataset in multilabel_datasets:
        print(f"Processing multi-label reliability diagram for {dataset}...")
        reliability_data = load_reliability_data(dataset, models)
        
        if reliability_data:
            plot_multilabel_reliability_diagram(reliability_data, dataset, output_dir)
            print(f"Generated multi-label reliability diagram for {dataset}")
        else:
            print(f"No reliability data found for {dataset}")
    
    # Generate performance tables
    generate_performance_tables(all_metrics_by_dataset, output_dir)

if __name__ == "__main__":
    run_integrated_analysis()
