import os
from config import Config
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
from conformalprediction.regression import ConformalRegressionPredictor
from utils import  *
from analysis.run_analysis import run_analysis
def seed_everything(seed=23):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def run_inference():
    # Set seed
    seed_everything(Config.SEED)
    
    # Setup device
    device = torch.device(0 if torch.cuda.is_available() else 'cpu')
    load_type = torch.float16 if Config.LOAD_TYPE == "float16" else torch.float32
    
    # Load tokenizer
    tokenizer = (LlamaTokenizer if Config.USE_LLAMA else AutoTokenizer).from_pretrained(Config.MODEL_NAME_OR_PATH)
    tokenizer.pad_token_id = 0
    tokenizer.bos_token_id = 1
    tokenizer.eos_token_id = 2
    tokenizer.padding_side = "left"
    
    # Load model
    model_config = AutoConfig.from_pretrained(Config.MODEL_NAME_OR_PATH)
    model = AutoModelForCausalLM.from_pretrained(
        Config.MODEL_NAME_OR_PATH,
        torch_dtype=load_type,
        config=model_config,
        device_map='auto'
    )
    
    if device == torch.device('cpu'):
        model.float()
    
    model.eval()
    print("Model loaded successfully")
    
    # Load and prepare data
    infer_data = pd.read_json(Config.INFER_FILE, lines=True)
    # Filter data based on ds_type
    infer_data = infer_data[infer_data['ds_type'] == Config.DS_TYPE]
    instruction_list = infer_data.apply(
        lambda row: pd.Series(
            {'instruction': f"Human: \n{row['instruction']}\n\nAssistant:\n"}
        ), axis=1
    )['instruction'].to_list()
    true_values = infer_data['output'].to_list()
    ds_type = infer_data['ds_type'].to_list()

    
    # Generate responses
    os.makedirs(os.path.dirname(Config.RESULTS_FILE), exist_ok=True)
    
    progress_bar = tqdm(total=len(instruction_list), desc="Generating responses")
    with open(Config.RESULTS_FILE, 'w', encoding="utf-8") as write_f:
        for i in range(0, len(instruction_list), Config.BATCH_SIZE):
            batch_data = instruction_list[i: min(i + Config.BATCH_SIZE, len(instruction_list))]
            batch_true_values = true_values[i: min(i + Config.BATCH_SIZE, len(true_values))]
            batch_ds_type = ds_type[i: min(i + Config.BATCH_SIZE, len(ds_type))]

            inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
            
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict_in_generate=True,
                output_logits=True,
                **Config.GENERATION_CONFIG
            )
            sequences = generation_output.sequences
            logits = generation_output.logits  # This is a tuple of tensors
            num_generated = len(logits)  # Number of generated tokens
            
            for j in range(sequences.shape[0]):  # Loop through each item in batch
                generated_tokens = sequences[j, -num_generated:]  # Get just the new tokens
                logits_generated_tokens = [logits[step][j] for step in range(num_generated)]  # Get full logits for each generated token
                #shape of logits_generated_tokens is (num_generated, vocab_size)
                if Config.DS_TYPE in Config.TASK_TYPES["ordinal_classification"]:
                    probs = get_probs(generated_tokens, logits_generated_tokens, tokenizer, Config.DS_TYPE)
                    # Convert tensor probabilities to Python floats
                    if probs is not None:
                        probs = [float(p.cpu()) for p in probs]
                else:
                    probs = None
            
                answer = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                data_one = {
                    "ds_type": batch_ds_type[j],
                    "input": batch_data[j],
                    "true_value": batch_true_values[j],
                    "prediction": answer,
                    "probs": probs,
                }
                if j == 0:
                    print(f"batch:{i} data: {data_one}")
                    print(f"*"*50)
                write_f.write(json.dumps(data_one, ensure_ascii=False) + "\n")
                progress_bar.update(1)  # Update progress bar for each processed item
                
                if Config.VERBOSE:  # Add verbose flag to control detailed output
                    print(f"Input type: {batch_ds_type[j]}")
                    print(f"True value: {batch_true_values[j]}")
                    print(f"Generated response {i+j+1}/{len(instruction_list)}:")
                    print(f"Response: {answer}")
                    print("Generated tokens and their logits:")
                    
                    for token, logit in zip([tokenizer.decode(t) for t in generated_tokens], logits_generated_tokens):
                        print(f"Token: '{token}', Logit: {logit:.4f}")
                    print("-" * 50)
    
    progress_bar.close()
def run_conformal_prediction(dataset_type):
    #load results
    
    with open(Config.RESULTS_FILE, 'r', encoding="utf-8") as read_f:
        results = [json.loads(line) for line in read_f]
    #filter results using DS_TYPE
    results = [result for result in results if result["ds_type"] == dataset_type]
    #shuffle results and divide into calibration and test set using CALIBRATION_RATE
    results = cleaning_results(results, dataset_type)
    random.shuffle(results)
    calibration_size = int(len(results) * Config.CALIBRATION_RATE)
    true_calibration = [result["true_value"] for result in results[:calibration_size]]
    pred_calibration = [result["prediction"] for result in results[:calibration_size]]
    probs_calibration = [result["probs"] for result in results[:calibration_size]]

    input_test = [result["input"] for result in results[calibration_size:]]
    true_test = [result["true_value"] for result in results[calibration_size:]]
    pred_test = [result["prediction"] for result in results[calibration_size:]]
    probs_test = [result["probs"] for result in results[calibration_size:]]
    
    #Convert predictions and true values to touple of class and keyword if it is classification
    if dataset_type in Config.TASK_TYPES['ordinal_classification']:
        true_calibration = get_prediction_touples(true_calibration, dataset_type)
        pred_calibration = get_prediction_touples(pred_calibration,dataset_type)
        #true_calibration = [int(prediction.strip().split(":")[0].strip()) for prediction in true_calibration]
        true_test = get_prediction_touples(true_test, dataset_type)
        pred_test = get_prediction_touples(pred_test,dataset_type)


    #compute conformal prediction
    baseline_cp = get_predictor(dataset_type)
    for alpha in Config.CP_ALPHA:
        q_hat = baseline_cp.fit(true_calibration, pred_calibration,probs_calibration, alpha)
        conformal_results = baseline_cp.get_conformal_results(true_test,pred_test, probs_test, q_hat)
        
        print(f"Confidence: {1-alpha:.2f} Coverage: {conformal_results[1]:.3f}  Size: {conformal_results[2]:.2f}")
        save_cp_results(dataset_type, input_test, true_test, pred_test, probs_test, conformal_results, alpha)

if __name__ == "__main__":
    #1: Get model responses
    run_inference()
    #2: Get conformal prediction results
    #dataset_type = Config.DS_TYPE
    #run_conformal_prediction(dataset_type)
    #3: Analyze results
    #run_analysis()
