import json
import numpy as np
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from analysis.analysis_utils import get_performance_metrics
from analysis.visualization_utils import calibration_diagram

def run_analysis():
    try:
        # Load the single results file containing all alpha values
        with open(Config.CONFORMAL_RESULTS_FILE, 'r', encoding="utf-8") as read_f:
            all_results = json.load(read_f)
            
        # Store results for all alpha values
        all_metrics = {}
        
        # Process each alpha value
        first_alpha_metrics = None  # Store metrics for first alpha
        
        for alpha in [Config.CP_ALPHA[0]]:
            alpha_str = str(alpha)
            if alpha_str not in all_results:
                print(f"Warning: No results found for α={alpha}")
                continue
                
            print(f"\nAnalyzing results for α={alpha}:")
            results = all_results[alpha_str]
            
            metrics = get_performance_metrics(results, Config.DS_TYPE, Config.TASK_TYPES)
            all_metrics[alpha_str] = metrics
            
            # Store first alpha metrics if not already stored
            if first_alpha_metrics is None:
                first_alpha_metrics = metrics
            
            if Config.DS_TYPE in Config.TASK_TYPES["regression"]:
                print("\nRegression Analysis Results:")
                print(f"Pearson Correlation Coefficient: {metrics['pearson_correlation']:.4f}")
                print(f"P-value: {metrics['p_value']:.4f}")
                print(f"Mean Absolute Error: {metrics['mae']:.4f}")
                print(f"Root Mean Square Error: {metrics['rmse']:.4f}")
                print(f"R² Score: {metrics['r2']:.4f}")
                
                # Generate calibration plot
                output_dir = Config.PLOTS_DIR
                plot_metrics = calibration_diagram(
                    results,
                    Config.DS_TYPE,
                    alpha,
                    output_dir=output_dir
                )
                
                print("\nCalibration Plot Results:")
                print(f"Coverage: {plot_metrics['coverage']:.3f}")
                print(f"Average Interval Size: {plot_metrics['avg_interval_size']:.3f}")
                print(f"Points within interval: {plot_metrics['in_interval_count']}/{plot_metrics['total_points']}")
                
        
            elif Config.DS_TYPE in Config.TASK_TYPES["classification"]:
                print("\nClassification Analysis Results:")
                # TODO: Add classification metrics display
                pass
            elif Config.DS_TYPE in Config.TASK_TYPES["ordinal_classification"]:
                # Only print metrics for first alpha
                if metrics == first_alpha_metrics:
                    print("\nOrdinal Classification Analysis Results:")
                    print(f"Accuracy (acc): {metrics['accuracy']:.4f}")
                    print(f"Micro-F1 (mi-F1): {metrics['micro_f1']:.4f}")
                    print(f"Macro-F1 (ma-F1): {metrics['macro_f1']:.4f}")
                    print(f"Macro-Average Pearson (ave): {metrics['macro_average']:.4f}")
                    
                    print("\nPer-Emotion Pearson Correlation:")
                    for emotion, pearson in metrics['per_emotion_pearson'].items():
                        print(f"{emotion}: {pearson:.4f}")
                    output_dir = Config.PLOTS_DIR
                    plot_metrics = calibration_diagram(
                    results,
                    Config.DS_TYPE,
                    alpha,
                    output_dir=output_dir
                )
                    
            elif Config.DS_TYPE in Config.TASK_TYPES["multiclass_classification"]:
                print("\nMulticlass Classification Analysis Results:")
                # TODO: Add multiclass classification metrics display
                pass
        
        # Save combined metrics for all alpha values
        metrics_file = os.path.join(Config.PLOTS_DIR, 'combined_metrics.json')
        os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
        with open(metrics_file, 'w') as f:
            json.dump(all_metrics, f, indent=2)
        print(f"\nCombined metrics saved to: {metrics_file}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in results file: {e}")
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        raise  # Add this to see the full traceback

    