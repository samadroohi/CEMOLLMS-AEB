# CEMOLLMS-AEB
Run_inference()
run calibration()
run test_results ()
run calibration_analysis.py
Running calibration_plots.py
d:; cd d:\samadfolder\Python\CEMOLLMS-AEB; .\.venv\Scripts\python.exe calibration_plots.py --dataset GoEmotions --model Emollama-7b
/media/samad/projects/phd/CEMOLLMS-AEB/.venv/bin/python -m src.analysis.generate_performance_tables --temperature temp_0.9

To get calibration plots for baseline platt and isotonic regression run calibration_baseline.py

To get all plots for calibration of conformal pred run run_all_calibration_plots.py
To get all tables for calibraiton of conformal prediciton run run_all_calibration_tables.py