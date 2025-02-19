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
                return_dict_in_generate=True,
                output_scores=True,
                **Config.GENERATION_CONFIG
            )
            
            sequences = generation_output.sequences
            logits = generation_output.scores  # Original logits from the model
            
            for j in range(sequences.shape[0]):
                response = tokenizer.decode(sequences[j], skip_special_tokens=True)
                
                # Store both full logits and probabilities for each token position
                token_info = []
                for step_logits in logits:
                    # Get logits for current sequence
                    seq_logits = step_logits[j]  # Shape: [vocab_size]
                    
                    # Get top k logits and their indices
                    top_k = 5  # Adjust this number to get more or fewer top tokens
                    top_logits, top_indices = seq_logits.topk(top_k)
                    
                    # Convert to probabilities
                    probs = seq_logits.softmax(dim=-1)
                    top_probs = probs[top_indices]
                    
                    # Get corresponding tokens
                    top_tokens = [tokenizer.decode([idx.item()]) for idx in top_indices]
                    
                    # Store information for this position
                    step_info = {
                        "top_tokens": top_tokens,
                        "logits": top_logits.tolist(),
                        "probabilities": top_probs.tolist(),
                    }
                    token_info.append(step_info)
                
                data_one = {
                    "output": response,
                    "token_info": token_info
                }
                write_f.write(json.dumps(data_one, ensure_ascii=False) + "\n")
                
                # Print example output for the first token
                print(f"Generated response {i+j+1}/{len(instruction_list)}:")
                print(f"Response: {response}")
                print(f"First token details:")
                print(f"Top tokens: {token_info[0]['top_tokens']}")
                print(f"Logits: {token_info[0]['logits']}")
                print(f"Probabilities: {token_info[0]['probabilities']}")
                print("-" * 50)

if __name__ == "__main__":
    run_inference()
