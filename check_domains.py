#!/usr/bin/env python3
"""
Script to check prediction domains for each model and dataset.
"""

import json
import os
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DomainChecker:
    def __init__(self, results_dir: str = "results/responses"):
        self.results_dir = Path(results_dir)
        self.datasets = ["EI-reg", "V-reg", "SST", "V-A,V-M,V-NYT,V-T"]
        self.models = ["Emobloom-7b", "Emollama-7b", "Emollama-chat-13b", "Emollama-chat-7b", "Emoopt-13b"]
        self.temp_dir = "temp_0.9"
        
        # Define expected domain ranges
        self.expected_domains = {
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
    
    def check_dataset_domains(self, dataset: str) -> Dict:
        """Check prediction domains for a specific dataset."""
        print(f"\n{'='*60}")
        print(f"Checking domains for dataset: {dataset}")
        print(f"Expected domain: {self.expected_domains[dataset]}")
        print(f"{'='*60}")
        
        domain_stats = {}
        
        for model in self.models:
            file_path = self.results_dir / dataset / self.temp_dir / f"{model}.json"
            
            if not file_path.exists():
                print(f"Warning: File not found: {file_path}")
                continue
                
            predictions = []
            true_values = []
            out_of_domain_count = 0
            total_count = 0
            
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.strip():
                            data = json.loads(line.strip())
                            
                            # Parse true value
                            try:
                                true_val = float(data['true_value'])
                                true_values.append(true_val)
                            except (ValueError, KeyError):
                                continue
                                
                            # Parse and clean prediction
                            pred_val = self.clean_prediction_value(data.get('prediction'))
                            if pred_val is None:
                                continue
                                
                            predictions.append(pred_val)
                            total_count += 1
                            
                            # Check if prediction is within expected domain
                            expected_min, expected_max = self.expected_domains[dataset]
                            if pred_val < expected_min or pred_val > expected_max:
                                out_of_domain_count += 1
                
                if predictions:
                    pred_array = np.array(predictions)
                    true_array = np.array(true_values)
                    
                    domain_stats[model] = {
                        'total_predictions': total_count,
                        'out_of_domain': out_of_domain_count,
                        'out_of_domain_pct': (out_of_domain_count / total_count) * 100,
                        'pred_min': np.min(pred_array),
                        'pred_max': np.max(pred_array),
                        'pred_mean': np.mean(pred_array),
                        'pred_std': np.std(pred_array),
                        'true_min': np.min(true_array),
                        'true_max': np.max(true_array),
                        'true_mean': np.mean(true_array),
                        'true_std': np.std(true_array)
                    }
                    
                    # Print detailed stats
                    print(f"\n{model}:")
                    print(f"  Total predictions: {total_count}")
                    print(f"  Out of domain: {out_of_domain_count} ({domain_stats[model]['out_of_domain_pct']:.1f}%)")
                    print(f"  Prediction range: [{domain_stats[model]['pred_min']:.3f}, {domain_stats[model]['pred_max']:.3f}]")
                    print(f"  True value range: [{domain_stats[model]['true_min']:.3f}, {domain_stats[model]['true_max']:.3f}]")
                    print(f"  Prediction mean ± std: {domain_stats[model]['pred_mean']:.3f} ± {domain_stats[model]['pred_std']:.3f}")
                    print(f"  True value mean ± std: {domain_stats[model]['true_mean']:.3f} ± {domain_stats[model]['true_std']:.3f}")
                    
                    # Flag problematic models
                    if domain_stats[model]['out_of_domain_pct'] > 10:
                        print(f"  ⚠️  WARNING: {domain_stats[model]['out_of_domain_pct']:.1f}% of predictions are out of domain!")
                    elif domain_stats[model]['out_of_domain_pct'] > 0:
                        print(f"  ⚠️  CAUTION: {domain_stats[model]['out_of_domain_pct']:.1f}% of predictions are out of domain")
                    else:
                        print(f"  ✅ All predictions within domain")
                        
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                
        return domain_stats
    
    def check_all_domains(self):
        """Check domains for all datasets."""
        print("Domain Consistency Check for All Datasets and Models")
        print("="*80)
        
        all_stats = {}
        for dataset in self.datasets:
            all_stats[dataset] = self.check_dataset_domains(dataset)
        
        # Summary table
        print(f"\n{'='*80}")
        print("SUMMARY TABLE - Out of Domain Predictions")
        print(f"{'='*80}")
        
        summary_data = []
        for dataset in self.datasets:
            for model in self.models:
                if model in all_stats[dataset]:
                    stats = all_stats[dataset][model]
                    summary_data.append({
                        'Dataset': dataset,
                        'Model': model,
                        'Out_of_Domain_%': stats['out_of_domain_pct'],
                        'Pred_Range': f"[{stats['pred_min']:.2f}, {stats['pred_max']:.2f}]",
                        'Expected_Range': str(self.expected_domains[dataset])
                    })
        
        df = pd.DataFrame(summary_data)
        print(df.to_string(index=False))
        
        # Save detailed results
        df.to_csv("domain_check_results.csv", index=False)
        print(f"\nDetailed results saved to: domain_check_results.csv")
        
        return all_stats

def main():
    """Main function to run domain checking."""
    checker = DomainChecker()
    checker.check_all_domains()

if __name__ == "__main__":
    main()
