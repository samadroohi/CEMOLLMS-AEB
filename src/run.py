import os

from sklearn.preprocessing import scale
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
from analysis.merged_analysis import run_integrated_analysis
import gc
import gc, re
import numpy as np
import torch
from math import ceil
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import GradientBoostingRegressor
from numpy.lib.format import open_memmap

#Helpers

def load_embeddings_any(path, results, dtype=np.float16):
    """
    Try np.load with mmap (works if the file was created with open_memmap / .npy header).
    If that fails, infer shape for a raw memmap using:
      N = max(row_index) + 1
      D = (filesize_bytes / itemsize) / N
    Returns: E (array-like, memmapped), N, D
    """
    # 1) Try headered .npy
    try:
        E = np.load(path, mmap_mode='r')
        if E.ndim != 2:
            raise ValueError(f"Expected 2D embeddings, got shape={E.shape}")
        N, D = E.shape
        return E, N, D
    except Exception:
        pass  # fall through

    # 2) Headerless raw memmap: infer N, D
    if not results:
        raise ValueError("Cannot infer N without results; results is empty.")

    N = max(r["row_index"] for r in results) + 1
    itemsize = np.dtype(dtype).itemsize
    fbytes = os.path.getsize(path)

    if fbytes % (itemsize * N) != 0:
        raise ValueError(
            f"File size {fbytes} not divisible by itemsize*N = {itemsize}*{N}. "
            "If this is a .npy, keep it headered; otherwise ensure dtype/rows are correct."
        )

    D = fbytes // (itemsize * N)
    if D <= 0:
        raise ValueError("Inferred D <= 0; check embedding file / dtype / results row_index.")

    E = np.memmap(path, dtype=dtype, mode='r', shape=(N, D))
    return E, int(N), int(D)

def l2norm_rows(X):
    X = X.astype(np.float32, copy=False)
    n = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    return X / n


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    v = values[order]; w = weights[order]
    w = w / (w.sum() + 1e-12)
    cw = np.cumsum(w)
    idx = np.searchsorted(cw, q, side="left")
    return v[min(idx, len(v)-1)]

def gaussian_weights_from_cosine_dists(dists):
    # With unit vectors: cos = 1 - d^2/2, angular distance ~ sqrt(2 - 2cos)
    s = 1 - (dists**2)/2.0
    ang = np.sqrt(np.maximum(2 - 2*s, 0.0))
    h = np.median(ang) + 1e-12
    return np.exp(-(ang**2)/(2*h*h))


def parse_first_float_0_1(text):
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', text)
    if not m:
        return None
    v = float(m.group(0))
    return float(np.clip(v, 0.0, 1.0))

@torch.no_grad()
def pooled_hidden_from_inputs(model, input_ids, attention_mask, is_seq2seq: bool):
    """
    Returns mean-pooled last hidden state for the *input tokens only*.
    Works for:
      - causal LMs (AutoModelForCausalLM)
      - seq2seq encoders (T5/BART)
    Shape: [B, d]
    """
    if is_seq2seq:
        # T5/BART: use encoder
        encoder = model.get_encoder() if hasattr(model, "get_encoder") else model.model.encoder
        out = encoder(input_ids=input_ids,
                      attention_mask=attention_mask,
                      output_hidden_states=True,
                      return_dict=True)
        H = out.hidden_states[-1]  # [B, T, d]
    else:
        # causal: forward on inputs only (no generation)
        out = model(input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=True,
                    use_cache=False,
                    return_dict=True)
        H = out.hidden_states[-1]  # [B, T, d]

    mask = attention_mask.unsqueeze(-1).to(H.dtype)  # [B, T, 1]
    num = (H * mask).sum(dim=1)                      # [B, d]
    den = mask.sum(dim=1).clamp_min(1)
    pooled = (num / den)
    return pooled

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
    
    # Load model (causal vs seq2seq)
    model_config = AutoConfig.from_pretrained(Config.MODEL_NAME_OR_PATH)
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
    infer_data = infer_data[infer_data['ds_type'] == Config.DS_TYPE]
    instruction_list = infer_data.apply(
        lambda row: pd.Series({'instruction': f"Human: \n{row['instruction']}\n\nAssistant:\n"}), axis=1
    )['instruction'].to_list()
    true_values = infer_data['output'].to_list()
    ds_type = infer_data['ds_type'].to_list()

    # --------- NEW: prepare embedding memmap ----------
    N = len(instruction_list)
    os.makedirs(os.path.dirname(Config.RESULTS_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(Config.HIDDEN_OUT), exist_ok=True)
    
    # Probe hidden dim d with a tiny batch
    probe_batch = instruction_list[:min(max(1, Config.BATCH_SIZE), N)]
    probe_inputs = tokenizer(probe_batch, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.inference_mode():
        probe_emb = pooled_hidden_from_inputs(model, probe_inputs.input_ids, probe_inputs.attention_mask, is_seq2seq)
    d = probe_emb.shape[-1]
    
    # Create memmap (float16 to save space); optional L2 normalize later
    emb_map = open_memmap(Config.HIDDEN_OUT, mode='w+', dtype=np.float16, shape=(N, D))

    write_cursor = 0
    del probe_inputs, probe_emb
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Generate responses
    progress_bar = tqdm(total=len(instruction_list), desc="Generating responses")
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

                # Tokenize once for both generation + hidden extraction
                inputs = tokenizer(batch_data, return_tensors="pt", padding=True, truncation=True)
                input_ids = inputs.input_ids.to(device)
                attention_mask = inputs.attention_mask.to(device)

                with torch.inference_mode():
                    # ---- Hidden states for the inputs (BEFORE generation) ----
                    pooled = pooled_hidden_from_inputs(model, input_ids, attention_mask, is_seq2seq)  # [B, d]
                    pooled_cpu = pooled.detach().cpu().float().numpy()
                    if getattr(Config, "NORMALIZE_EMB", False):
                        norms = np.linalg.norm(pooled_cpu, axis=1, keepdims=True) + 1e-12
                        pooled_cpu = pooled_cpu / norms
                    emb_map[write_cursor:write_cursor + pooled_cpu.shape[0]] = pooled_cpu.astype(np.float16)

                    # ---- Generation ----
                    if is_seq2seq:
                        generation_output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=True,
                            output_scores=True,
                            **Config.GENERATION_CONFIG
                        )
                        logits_tuple = tuple(generation_output.scores) if hasattr(generation_output, 'scores') else tuple()
                    else:
                        generation_output = model.generate(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            return_dict_in_generate=True,
                            output_logits=True,
                            **Config.GENERATION_CONFIG
                        )
                        logits_tuple = tuple(logit.detach().cpu() for logit in generation_output.logits)

                cpu_sequences = generation_output.sequences.detach().cpu()
                num_generated = len(logits_tuple) if logits_tuple else cpu_sequences.shape[1] - input_ids.shape[1]
                
                # free GPU
                del input_ids, attention_mask, generation_output, pooled
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
                for j in range(cpu_sequences.shape[0]):
                    response = None
                    generated_tokens = cpu_sequences[j, -num_generated:]
                    logits_generated_tokens = [logits_tuple[step][j] for step in range(num_generated)]
                    
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

                    rec = {
                        "ds_type": batch_ds_type[j],
                        "input": batch_data[j],
                        "true_value": batch_true_values[j],
                        "prediction": response,
                        "probs": probs,
                        # NEW: row index to align with embedding file
                        "row_index": write_cursor + j
                    }
                    if getattr(Config, "SAVE_VALENCE", False):
                        v = parse_first_float_0_1(response)
                        if v is not None:
                            rec["valence"] = float(v)

                    if results_written < 5:
                        print(f"First result data: {rec}")
                        print("*" * 50)
                    
                    write_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    progress_bar.update(1)
                    results_written += 1

                    # free per-item
                    del generated_tokens, logits_generated_tokens
                
                # advance pointers
                i = end_idx
                write_cursor += pooled_cpu.shape[0]

                # cleanup
                del cpu_sequences, logits_tuple, pooled_cpu, inputs
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            except torch.cuda.OutOfMemoryError as e:
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
    
    # Flush memmap
    del emb_map
    print(f"Saved pooled hidden states → {Config.HIDDEN_OUT} with shape=({N}, {d}), dtype=float16")
    
    # cleanup
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

def run_conformal_prediction():
    dataset_type = Config.DS_TYPE

    # Load & filter results
    with open(Config.RESULTS_FILE, 'r', encoding="utf-8") as read_f:
        all_rows = [json.loads(line) for line in read_f]
    results = [r for r in all_rows if r["ds_type"] == dataset_type]

    # Clean/shuffle/split (your function)
    results = cleaning_results(results, dataset_type)
    random.shuffle(results)
    calibration_size = int(len(results) * Config.CALIBRATION_RATE)

    # --- Load embeddings robustly (works for .npy or raw) ---
    E, N, D = load_embeddings_any(Config.HIDDEN_OUT, results, dtype=np.float16)

    # Build lists
    true_calibration = [r["true_value"] for r in results[:calibration_size]]
    pred_calibration = [r["prediction"] for r in results[:calibration_size]]
    probs_calibration = [r["probs"] for r in results[:calibration_size]]
    idx_calibration  = [r["row_index"] for r in results[:calibration_size]]

    input_test = [r["input"] for r in results[calibration_size:]]
    true_test  = [r["true_value"] for r in results[calibration_size:]]
    pred_test  = [r["prediction"] for r in results[calibration_size:]]
    probs_test = [r["probs"] for r in results[calibration_size:]]
    idx_test   = [r["row_index"] for r in results[calibration_size:]]

    # Pull embeddings for cal/test
    X_cal = np.stack([E[i].astype(np.float32) for i in idx_calibration], axis=0)
    X_test = np.stack([E[i].astype(np.float32) for i in idx_test], axis=0)

    # L2-normalize for cosine geometry
    X_cal = l2norm_rows(X_cal)
    X_test = l2norm_rows(X_test)

    # Parse numeric centers (valence) once; make sure they exist in your JSONL
    center_calibration = np.array([r.get("valence", None) for r in results[:calibration_size]], dtype=float)
    center_test = np.array([r.get("valence", None) for r in results[calibration_size:]], dtype=float)
    if np.any(np.isnan(center_calibration)) or np.any(np.isnan(center_test)):
        raise ValueError("Missing numeric 'valence' in results. Save it during inference or parse it here.")

    y_cal = np.asarray(true_calibration, dtype=np.float32)
    y_test = np.asarray(true_test, dtype=np.float32)
    m_cal = np.asarray(center_calibration, dtype=np.float32)
    m_test = np.asarray(center_test, dtype=np.float32)

    # For classification branches you had; left as-is
    if dataset_type in Config.TASK_TYPES['ordinal_classification'] or dataset_type in Config.TASK_TYPES['multiclass_classification']:
        tuples_calibration = get_prediction_touples(true_calibration, pred_calibration, probs_calibration, dataset_type)
        true_calibration, pred_calibration, probs_calibration = tuples_calibration
        tuples_test = get_prediction_touples(true_test, pred_test, probs_test, dataset_type)
        true_test, pred_test, probs_test = tuples_test
    # 1) Basic counts
    print("n_cal:", len(X_cal), "n_test:", len(X_test))

    # 2) Residual stats on calibration
    r_cal = np.abs(y_cal - m_cal)
    print("r_cal: mean,median,std,min,max", r_cal.mean(), np.median(r_cal), r_cal.std(), r_cal.min(), r_cal.max())

   
    # --- Conformal prediction ---
    for alpha in Config.CP_ALPHA:
        for ttype in Config.TASK_TYPES:
            if Config.DS_TYPE in Config.TASK_TYPES[ttype]:
                baseline_cp = get_predictor(ttype, alpha)  # should return your LocalWeightedCPredictor
                if ttype == "weighted_regression" or ttype == "local_regression":
                    #TBD: Use local clustered CP
                    pass
                else:
                    q_hat = baseline_cp.fit(true_calibration, pred_calibration, probs_calibration, alpha)
                    conformal_results = baseline_cp.get_conformal_results(true_test, pred_test, probs_test, q_hat)

                    print(f" Task: {ttype} Confidence: {1-alpha:.2f} Coverage: {conformal_results[1]:.3f}  Size: {conformal_results[2]:.2f}")
                    save_cp_results(dataset_type, input_test, true_test, pred_test, probs_test, conformal_results, alpha)
        else:
            print(f"Dataset type {Config.DS_TYPE} not found in TASK_TYPES; skipping conformal prediction.")

if __name__ == "__main__":
    analysis = False # False if you want to run inference and conformal prediction
    model_names = [
        "lzw1008/Emollama-7b",
        #"lzw1008/Emollama-chat-7b",
        #"lzw1008/Emollama-chat-13b",
        #"lzw1008/Emoopt-13b",
        #"lzw1008/Emobloom-7b",
        #"lzw1008/Emot5-large", 
        #"lzw1008/Emobart-large"
    ]
    dataset_names = [
        #"EI-oc", 
        #"TDT", 
        #"SST5",
        #"V-oc",  
        "EI-reg", 
        "V-reg", 
        "V-A,V-M,V-NYT,V-T", 
        "Emobank", 
        "SST", 
        #"GoEmotions", 
        #"E-c"
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
                run_conformal_prediction()
                
                # Clear GPU cache after processing each dataset
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print(f"GPU memory cleared after processing {dataset_name}")
                    
                # Force Python garbage collection
                gc.collect()
                
                print(f"Completed processing for {model_name} on {dataset_name}")
                print("-" * 50)
            #run_analysis(model_name, dataset_name)
    
    # Run integrated analysis after all individual analyses are complete
    #print("\nRunning integrated analysis...")
    #run_integrated_analysis()
    #print("Integrated analysis completed!")
