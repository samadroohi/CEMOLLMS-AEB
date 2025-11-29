# 🔍 Conformal Prediction for Affect Recognition

This repository provides an end-to-end pipeline for performing inference, conformal prediction calibration, and analysis on affective emotion recognition. It includes scripts to generate plots and result tables to evaluate calibration performance.

> 📌 **Publication Status:**  
> The related paper is currently under review in *SPJ Intelligent Computing*.
---

## 🚀 How to Run the Project

Follow the steps below to execute the full pipeline and generate results.

### 1️⃣ Run Inference + Conformal Prediction

Open `run.py` and **uncomment** the following function calls:

- `run_inference`
- `run_conformal_prediction`

Then execute:

```bash
python run.py

### 2️⃣ Do initial analysis
To get initial analysis results you should uncomment run_analysis() before running run.py 

Uncomment `run_analysis()` in `run.py`, then run:

```bash
python run.py
```

### 3️⃣ Complete analysis

To get calibration plots for baseline platt and isotonic regression run calibration_baseline.py
To get all plots for calibration of conformal pred run run_all_calibration_plots.py
To get all tables for calibraiton of conformal prediciton run run_all_calibration_tables.py
 To get adaptiveness of CQR python src/analysis/interval_width_diagnostics.py --alpha 0.5 (for alphas 0.1 to 0.5)

### 4️⃣ Installation & Requirements
Create a virtual environment and install dependencies:
torch
matplotlib
scikit-learn
accelerate
bitsandbytes
pandas
seaborn
transformers
