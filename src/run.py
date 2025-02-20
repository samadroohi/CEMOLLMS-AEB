import os
from config import Config
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json
import random
import numpy as np
import argparse

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
    responses = []
    os.makedirs(os.path.dirname(Config.PREDICT_FILE), exist_ok=True)
    
    with open(Config.PREDICT_FILE, 'w', encoding="utf-8") as write_f:
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
                response = tokenizer.decode(sequences[j], skip_special_tokens=True)
                generated_tokens = sequences[j, -num_generated:]  # Get just the new tokens
                
                # Get logits for this batch item
                token_logits = []
                for step in range(num_generated):
                    # Get logit for the token that was actually generated
                    generated_token = generated_tokens[step]
                    token_logits.append(logits[step][j][generated_token].item())
                
                data_one = {
                    "output": response,
                    "token_logits": token_logits,
                    "tokens": [tokenizer.decode(t) for t in generated_tokens],
                    "true_value": batch_true_values[j],
                    "ds_type": batch_ds_type[j]
                }
                write_f.write(json.dumps(data_one, ensure_ascii=False) + "\n")
                responses.append(response)
                print(f"Input type: {batch_ds_type[j]}")
                print(f"True value: {batch_true_values[j]}")
                print(f"Generated response {i+j+1}/{len(instruction_list)}:")
                print(f"Response: {response}")
                print("Generated tokens and their logits:")
                
                for token, logit in zip([tokenizer.decode(t) for t in generated_tokens], token_logits):
                    print(f"Token: '{token}', Logit: {logit:.4f}")
                print("-" * 50)

if __name__ == "__main__":
    run_inference()
