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
    instruction_list = infer_data.apply(
        lambda row: pd.Series(
            {'instruction': f"Human: \n{row['instruction']}\n\nAssistant:\n"}
        ), axis=1
    )['instruction'].to_list()
    
    # Generate responses
    responses = []
    os.makedirs(os.path.dirname(Config.PREDICT_FILE), exist_ok=True)
    
    with open(Config.PREDICT_FILE, 'w', encoding="utf-8") as write_f:
        for i in range(0, len(instruction_list), Config.BATCH_SIZE):
            batch_data = instruction_list[i: min(i + Config.BATCH_SIZE, len(instruction_list))]
            inputs = tokenizer(batch_data, return_tensors="pt", padding=True)
            
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            
            generation_output = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                **Config.GENERATION_CONFIG
            )
            
            for j in range(generation_output.shape[0]):
                response = tokenizer.decode(generation_output[j], skip_special_tokens=True)
                data_one = {"output": response}
                write_f.write(json.dumps(data_one, ensure_ascii=False) + "\n")
                responses.append(response)
                print(f"Generated response {i+j+1}/{len(instruction_list)}:")
                print(response)
                print("-" * 50)

if __name__ == "__main__":
    run_inference()
