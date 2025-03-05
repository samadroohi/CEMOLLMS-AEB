import json
import numpy as np
import sys
import os
from analysis.visualization_utils import calibration_anlaysis

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from analysis.analysis_utils import get_performance_metrics
from analysis.visualization_utils import calibration_anlaysis

def run_analysis(model_name, dataset_name):
    try:
        Config.update_model_and_dataset(model_name, dataset_name)
        # Load the single results file containing all alpha values
        with open(Config.CONFORMAL_RESULTS_FILE, 'r', encoding="utf-8") as read_f:
            results = json.load(read_f)

        performance_metrics = get_performance_metrics(results[str(Config.CP_ALPHA[0])], Config.DS_TYPE, Config.TASK_TYPES)
        
        # Initialize these variables before the if-elif blocks
        calibration_metrics = {}
        cp_metrics = {}
        
        if Config.DS_TYPE in Config.TASK_TYPES["regression"]:
            print("\nRegression Analysis Results:")
            print(f"Pearson Correlation Coefficient: {performance_metrics['pearson_correlation']:.4f}")
            print(f"P-value: {performance_metrics['p_value']:.4f}")
            print(f"Mean Absolute Error: {performance_metrics['mae']:.4f}")
            print(f"Root Mean Square Error: {performance_metrics['rmse']:.4f}")
            print(f"R² Score: {performance_metrics['r2']:.4f}")
            
            # Generate calibration plot
            output_dir = Config.PLOTS_DIR
            calibration_metrics, cp_metrics = calibration_anlaysis(
                results,
                Config.DS_TYPE,
                output_dir=output_dir
            )
            
            print("\nCalibration Plot Results:")
 

            print ("calibration metrics: ", calibration_metrics)
            print ("cp_metrics: ", cp_metrics)

        elif Config.DS_TYPE in Config.TASK_TYPES["classification"]:
            print("\nClassification Analysis Results:")
            # TODO: Add classification metrics display
            pass
        elif Config.DS_TYPE in Config.TASK_TYPES["ordinal_classification"]:
            print("\nOrdinal Classification Analysis Results:")
            print(f"Accuracy (acc): {performance_metrics['accuracy']:.4f}")
            print(f"Micro-F1 (mi-F1): {performance_metrics['micro_f1']:.4f}")
            print(f"Macro-F1 (ma-F1): {performance_metrics['macro_f1']:.4f}")
            print(f"Pearson correlation coefficient:{performance_metrics['average_pearson']}")
            if performance_metrics['macro_average'] != None:
                print(f"Macro-Average Pearson (ave): {performance_metrics['macro_average']:.4f}")
            if performance_metrics['per_emotion_pearson'] != None:
                print("\nPer-Emotion Pearson Correlation:")
                for emotion, pearson in performance_metrics['per_emotion_pearson'].items():
                    print(f"{emotion}: {pearson:.4f}")
            output_dir = Config.PLOTS_DIR
            
            calibration_metrics, cp_metrics  = calibration_anlaysis(
            results,
            Config.DS_TYPE,
            output_dir=output_dir)

            print("\nCalibration Plot Results:")
            print(f"Accuracy: {calibration_metrics.get('accuracy', 'N/A')}")
            print(f"ECE: {calibration_metrics.get('ece', 'N/A')}")
            print(f"MCalE: {calibration_metrics.get('mcale', 'N/A')}")
            print(f"Brier Score: {calibration_metrics.get('brier_score', 'N/A')}")
            print(f"Alpha: {cp_metrics['alpha']} Coverage: {cp_metrics['coverage']} Prediction Set Size: {cp_metrics['psize']}, ACE: {cp_metrics['ace']}, MCE: {cp_metrics['mcove']}")
            # Option 1: Using update() method
            

        elif Config.DS_TYPE in Config.TASK_TYPES["multiclass_classification"]:
            print("\nMulticlass Classification Analysis Results:")
            print(f"Jaccard Index: {performance_metrics['jaccard_index']:.4f}")
            print(f"Micro-F1: {performance_metrics['f1_micro']:.4f}")
            print(f"Macro-F1: {performance_metrics['f1_macro']:.4f}")
            
            # If you want to generate calibration plots for multiclass
            output_dir = Config.PLOTS_DIR
            calibration_metrics, cp_metrics = calibration_anlaysis(
                results,
                Config.DS_TYPE,
                output_dir=output_dir
            )
        
        # Only save metrics if they're not empty
        if calibration_metrics or cp_metrics or performance_metrics:
            merged_metrics = performance_metrics.copy()  # Start with performance metrics
            # Add calibration and CP metrics if they exist
            if calibration_metrics:
                merged_metrics.update(calibration_metrics)
            if cp_metrics:
                merged_metrics.update(cp_metrics)
            
            metrics_file = os.path.join(Config.PLOTS_DIR, 'combined_metrics.json')
            os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
            with open(metrics_file, 'w') as f:
                json.dump(merged_metrics, f, indent=2)
            print(f"\nCombined metrics saved to: {metrics_file}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find results file: {e}")
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON format in results file: {e}")
    except Exception as e:
        print(f"Unexpected error during analysis: {e}")
        raise  # Add this to see the full traceback

    