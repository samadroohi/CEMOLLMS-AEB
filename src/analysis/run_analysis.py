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

def run_analysis():
    try:
        # Load the single results file containing all alpha values
        with open(Config.CONFORMAL_RESULTS_FILE, 'r', encoding="utf-8") as read_f:
            results = json.load(read_f)

        performance_metrics = get_performance_metrics(results[str(Config.CP_ALPHA[0])], Config.DS_TYPE, Config.TASK_TYPES)
        
        if Config.DS_TYPE in Config.TASK_TYPES["regression"]:
            print("\nRegression Analysis Results:")
            print(f"Pearson Correlation Coefficient: {performance_metrics['pearson_correlation']:.4f}")
            print(f"P-value: {performance_metrics['p_value']:.4f}")
            print(f"Mean Absolute Error: {performance_metrics['mae']:.4f}")
            print(f"Root Mean Square Error: {performance_metrics['rmse']:.4f}")
            print(f"R² Score: {performance_metrics['r2']:.4f}")
            
            # Generate calibration plot
            output_dir = Config.PLOTS_DIR
            calibration_metrics, _ = calibration_anlaysis(
                results,
                Config.DS_TYPE,
                output_dir=output_dir
            )
            
            print("\nCalibration Plot Results:")
            print(f"Coverage: {calibration_metrics['coverage']:.3f}")
            print(f"Average Interval Size: {calibration_metrics['avg_interval_size']:.3f}")
            print(f"Points within interval: {calibration_metrics['in_interval_count']}/{plot_metrics['total_points']}")
            
            all_metrics = calibration_metrics
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
            print(f"Accuracy:{calibration_metrics["accuracy"]}")
            print(f"ECE:{calibration_metrics["ece"]}")
            print(f"MCE:{calibration_metrics["mce"]}")
            print(f"Brier Score:{calibration_metrics["brier_score"]}")
            print(f"Alpha: {cp_metrics['alpha']} Coverage:{cp_metrics['coverage']} Prediction Set Size:{cp_metrics['psize']}")
            all_metrics = calibration_metrics + cp_metrics   
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

    