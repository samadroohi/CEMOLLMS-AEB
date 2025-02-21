import json
import numpy as np
import sys
import os

# Add parent directory to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config import Config
from analysisutils import analyze_regression_metrics

def run_analysis():
    # load conformal results
    with open(Config.CONFORMAL_RESULTS_FILE, 'r', encoding="utf-8") as read_f:
        results = [json.loads(line) for line in read_f]
    # filter results using DS_TYPE
    results = [result for result in results if result["ds_type"] == Config.DS_TYPE]
    # filter results using temperature
    if Config.DS_TYPE in Config.TASK_TYPES["regression"]:
        #Run accuracy performance analysis for regression task
        #visualize prediction intervals for conformal regression
        true_values = np.array(results[0]["true_values"])
        predictions = np.array(results[0]["predictions"])
        
        # Get regression metrics
        metrics = analyze_regression_metrics(true_values, predictions)
        
        print("\nRegression Analysis Results:")
        print(f"Pearson Correlation Coefficient: {metrics['pearson_correlation']:.4f}")
        print(f"P-value: {metrics['p_value']:.4f}")
        print(f"Mean Absolute Error: {metrics['mae']:.4f}")
        print(f"Root Mean Square Error: {metrics['rmse']:.4f}")
        print(f"R² Score: {metrics['r2']:.4f}")

if __name__ == "__main__":
    run_analysis()