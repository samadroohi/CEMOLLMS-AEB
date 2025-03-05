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
import gc

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
    tokenizer.pad_token_id = 0 if tokenizer.pad_token_id is None else tokenizer.pad_token_id
    tokenizer.bos_token_id = 1 if tokenizer.bos_token_id is None else tokenizer.bos_token_id
    tokenizer.eos_token_id = 2 if tokenizer.eos_token_id is None else tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    
    # Load model
    model_config = AutoConfig.from_pretrained(Config.MODEL_NAME_OR_PATH)
    
    # Determine model type and load appropriate model class
    model_type = model_config.__class__.__name__
    if model_type == "T5Config":
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            torch_dtype=load_type,
            config=model_config,
            device_map='auto'
        )
        is_seq2seq = True
    elif model_type == "BartConfig":
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            torch_dtype=load_type,
            config=model_config,
            device_map='auto'
        )
        is_seq2seq = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            torch_dtype=load_type,
            config=model_config,
            device_map='auto'
        )
        is_seq2seq = False
    
    if device == torch.device('cpu'):
        model.float()
    
    model.eval()
    print(f"***************Model {Config.MODEL_NAME_OR_PATH} loaded successfully***************")
    print(f"Model type: {model_type}, Using seq2seq generation: {is_seq2seq}")
    
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
    
    # Start with configured batch size but be prepared to reduce it
    current_batch_size = Config.BATCH_SIZE
    
    results_written = 0
    with open(Config.RESULTS_FILE, 'w', encoding="utf-8") as write_f:
        i = 0
        while i < len(instruction_list):
            try:
                end_idx = min(i + current_batch_size, len(instruction_list))
                batch_data = instruction_list[i:end_idx]
                batch_true_values = true_values[i:end_idx]
                batch_ds_type = ds_type[i:end_idx]

                # Process in even smaller chunks if needed
                inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
                
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)
                
                with torch.inference_mode():  # More efficient than no_grad for inference
                    if is_seq2seq:
                        # For seq2seq models like T5 and BART
                        generation_output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=True,
                            output_scores=True,  # T5/BART uses scores instead of logits
                            **Config.GENERATION_CONFIG
                        )
                        # Convert scores to logits format if needed for downstream processing
                        if hasattr(generation_output, 'scores'):
                            # Create a tuple of tensors mimicking the logits structure
                            logits_tuple = tuple(score for score in generation_output.scores)
                        else:
                            logits_tuple = tuple()
                    else:
                        # For causal LMs
                        generation_output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=True,
                            output_logits=True,
                            **Config.GENERATION_CONFIG
                        )
                        logits_tuple = tuple(logit.detach().cpu() for logit in generation_output.logits)
                
                # Extract and immediately move to CPU to free GPU memory
                cpu_sequences = generation_output.sequences.detach().cpu()
                num_generated = len(logits_tuple) if logits_tuple else cpu_sequences.shape[1] - input_ids.shape[1]
                
                # Explicitly delete GPU tensors
                del input_ids, attention_mask, generation_output
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for j in range(cpu_sequences.shape[0]):  # Loop through each item in batch
                    response = None
                    generated_tokens = cpu_sequences[j, -num_generated:]
                    logits_generated_tokens = [logits_tuple[step][j] for step in range(num_generated)]
                    
                    # Process one example at a time
                    if Config.DS_TYPE in Config.TASK_TYPES["ordinal_classification"]:
                        probs = get_probs(generated_tokens, logits_generated_tokens, tokenizer, Config.DS_TYPE)
                        if probs is not None:
                            probs = [float(p) for p in probs]
                    elif Config.DS_TYPE in Config.TASK_TYPES["multiclass_classification"]:
                        probs = get_probs(generated_tokens, logits_generated_tokens, tokenizer, Config.DS_TYPE)
                        response = get_response_multiclass(generated_tokens, logits_generated_tokens, tokenizer, Config.DS_TYPE)
                    else:
                        probs = None
                    
                    if response is None:
                        response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
                        
                    data_one = {
                        "ds_type": batch_ds_type[j],
                        "input": batch_data[j],
                        "true_value": batch_true_values[j],
                        "prediction": response,
                        "probs": probs,
                    }
                    
                    if results_written <5:
                        print(f"First result data: {data_one}")
                        print(f"*"*50)
                    
                    write_f.write(json.dumps(data_one, ensure_ascii=False) + "\n")
                    progress_bar.update(1)
                    results_written += 1
                    
                    # Free memory for each sample
                    del generated_tokens, logits_generated_tokens
                
                # Successfully processed batch, move to next batch
                i = end_idx
                
                # Clean up more memory
                del cpu_sequences, logits_tuple
                gc.collect()
                
            except torch.cuda.OutOfMemoryError as e:
                # If we get OOM, reduce batch size and try again
                if current_batch_size > 1:
                    current_batch_size = max(1, current_batch_size // 2)
                    print(f"\nOOM error! Reducing batch size to {current_batch_size} and retrying...")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
                else:
                    print(f"\nFatal OOM error even with batch size 1. Error: {e}")
                    raise
    
    progress_bar.close()
    
    # After processing is complete:
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_conformal_prediction():
    #load results
    dataset_type = Config.DS_TYPE
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
    #if dataset_type in Config.TASK_TYPES['ordinal_classification']:
    if dataset_type in Config.TASK_TYPES['ordinal_classification'] or dataset_type in Config.TASK_TYPES['multiclass_classification']:
        tuples_calibration = get_prediction_touples(true_calibration,pred_calibration,probs_calibration, dataset_type)
        true_calibration = tuples_calibration[0]
        pred_calibration = tuples_calibration[1]
        probs_calibration = tuples_calibration[2]
        tuples_test = get_prediction_touples(true_test, pred_test,probs_test, dataset_type)
        true_test = tuples_test[0]
        pred_test = tuples_test[1]
        probs_test = tuples_test[2]

    #elif dataset_type in Config.TASK_TYPES['multiclass_classification']:
     #   tuples_test = get_prediction_touples(true_test, pred_test, dataset_type)
      #  true_test = tuples_test[0]
       # pred_test = tuples_test[1]

    #compute conformal prediction
    baseline_cp = get_predictor(dataset_type)
    for alpha in Config.CP_ALPHA:
        q_hat = baseline_cp.fit(true_calibration, pred_calibration,probs_calibration, alpha)
        conformal_results = baseline_cp.getmerged_metrics_conformal_results(true_test,pred_test, probs_test, q_hat)
        
        print(f"Confidence: {1-alpha:.2f} Coverage: {conformal_results[1]:.3f}  Size: {conformal_results[2]:.2f}")
        save_cp_results(dataset_type, input_test, true_test, pred_test, probs_test, conformal_results, alpha)

if __name__ == "__main__":
    analysis = False # False if you want to run inference and conformal prediction
    model_names = [
        "lzw1008/Emollama-7b",
        "lzw1008/Emollama-chat-7b",
        "lzw1008/Emollama-chat-13b",
        "lzw1008/Emoopt-13b",
        "lzw1008/Emobloom-7b",
        #"lzw1008/Emot5-large", 
        #"lzw1008/Emobart-large"
    ]
    dataset_names = [
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
    for model_name in model_names:
        for dataset_name in dataset_names:
            if not analysis:
                #if model_name includes llama
                if "llama" in model_name:
                    Config.USE_LLAMA = True
                else:
                    Config.USE_LLAMA = False

                Config.update_model_and_dataset(model_name, dataset_name)
                
                #1: Get model responses
                #run_inference()
                #2: Get conformal prediction results
                #run_conformal_prediction()
                
                # Clear GPU cache after processing each dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory cleared after processing {dataset_name}")
                    
                # Force Python garbage collection
                gc.collect()
                
                print(f"Completed processing for {model_name} on {dataset_name}")
                print("-" * 50)
            run_analysis(model_name, dataset_name)
