import os
from collections import defaultdict
from pathlib import Path
from sklearn.preprocessing import scale
import torch
from transformers import LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM, AutoConfig
import pandas as pd
import json
import random
import numpy as np
import argparse
from tqdm import tqdm
from conformalprediction.conformalizedquantileregression import ConformalizedQuantileRegressionPredictor
from numpy.lib.format import open_memmap
from conformalprediction.quantileregression import QuantileRegressor
from utils import  *
from joblib import dump, load  # used for caching the quantile regressor
import time



if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))
    from config import Config
    from conformalprediction.regression import ConformalRegressionPredictor
    from utils import *
    from analysis.run_analysis import run_analysis
    from analysis.merged_analysis import run_integrated_analysis
else:
    from .config import Config
    from .conformalprediction.regression import ConformalRegressionPredictor
    from .utils import *
    from .analysis.run_analysis import run_analysis
    from .analysis.merged_analysis import run_integrated_analysis
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
    emb_map = open_memmap(Config.HIDDEN_OUT, mode='w+', dtype=np.float16, shape=(N, d))

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
                    #if regression, parse valence
                    if Config.DS_TYPE == "regression_task":
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
     # --- tiny helper: linear interpolation over a dict {tau: vector} ---
    def _interp(preds_dict, q_target, q_grid, allow_extrapolate=False):
        if q_target in preds_dict:
            return preds_dict[q_target]
        qs = np.array(sorted(q_grid))
        hi = int(np.searchsorted(qs, q_target, side="left"))
        if hi == 0:
            if not allow_extrapolate: raise ValueError("q_target below grid; extend grid.")
            return preds_dict[qs[0]]
        if hi == len(qs):
            if not allow_extrapolate: raise ValueError("q_target above grid; extend grid.")
            return preds_dict[qs[-1]]
        q0, q1 = float(qs[hi-1]), float(qs[hi])
        p0, p1 = preds_dict[q0], preds_dict[q1]
        w = (q_target - q0) / max(q1 - q0, 1e-12)
        return (1 - w) * p0 + w * p1

    def _augment_with_metadata(X, preds, centers):
        """Concatenate scalar metadata (LLM point predictions, optional valence) to embeddings."""
        extras = []
        preds_arr = np.asarray(preds, dtype=np.float32).reshape(-1, 1)
        if np.any(np.isnan(preds_arr)):
            fill = float(np.nanmean(preds_arr))
            if not np.isfinite(fill):
                fill = 0.0
            preds_arr = np.where(np.isnan(preds_arr), fill, preds_arr)
        extras.append(preds_arr)

        if centers is not None:
            centers = np.asarray(centers, dtype=np.float32).reshape(-1, 1)
            if np.any(np.isnan(centers)):
                fill = float(np.nanmean(centers))
                if not np.isfinite(fill):
                    fill = 0.0
                centers = np.where(np.isnan(centers), fill, centers)
            extras.append(centers)

        if extras:
            extra_cols = np.concatenate(extras, axis=1)
            return np.concatenate([X, extra_cols], axis=1)
        return X

    def _sanitize_features(X, split_name):
        """Remove NaNs/Infs (column-mean fill) and fail loudly if any remain."""
        X = np.asarray(X, dtype=np.float32)
        if not np.any(~np.isfinite(X)):
            return X

        X = X.copy()
        bad_mask = ~np.isfinite(X)
        if np.any(bad_mask):
            col_means = np.nanmean(X, axis=0)
            col_means[~np.isfinite(col_means)] = 0.0
            rows, cols = np.where(bad_mask)
            X[rows, cols] = col_means[cols]

        if np.any(~np.isfinite(X)):
            cols = np.unique(np.where(~np.isfinite(X))[1])
            raise ValueError(f"{split_name} features still non-finite after sanitization; columns={cols[:10]}")

        return X
    
    dataset_type = Config.DS_TYPE
    Config.update_paths()

    aggregated_results_by_path = defaultdict(list)

    # Load & filter results
    with open(Config.RESULTS_FILE, 'r', encoding="utf-8") as read_f:
        all_rows = [json.loads(line) for line in read_f]
    results = [r for r in all_rows if r["ds_type"] == dataset_type]

    # Clean results before repeated resampling
    results = cleaning_results(results, dataset_type)
    num_samples = len(results)
    train_size = int(num_samples * Config.TRAIN_SET_SIZE)
    calibration_size = int(num_samples * Config.CALIBRATION_SET_SIZE)
    allocated = train_size + calibration_size
    test_size = max(0, num_samples - allocated)
    if allocated + test_size != num_samples:
        test_size = num_samples - allocated

    # --- Load embeddings robustly (works for .npy or raw) ---
    E, N, D = load_embeddings_any(Config.HIDDEN_OUT, results, dtype=np.float16)

    is_multiclass_task = dataset_type in Config.TASK_TYPES.get("multiclass_classification", [])
    is_ordinal_task = dataset_type in Config.TASK_TYPES.get("ordinal_classification", [])
    is_regression_task = dataset_type in Config.TASK_TYPES.get("regression_tasks", [])

    original_multiclass_mode = getattr(Config, "MULTICLASS_CP_MODE", None)
    original_ordinal_mode = getattr(Config, "ORDINAL_CP_MODE", None)
    original_regression_mode = getattr(Config, "REGRESSION_CP_MODE", None)

    if is_multiclass_task:
        modes_to_run = Config.get_multiclass_modes()
    elif is_ordinal_task:
        modes_to_run = Config.get_ordinal_modes()
    elif is_regression_task:
        modes_to_run = Config.get_regression_modes()
    else:
        modes_to_run = [None]

    for repeat_idx in range(Config.NUM_REPEATS):
        print(f"\n--- Conformal Prediction Repeat {repeat_idx + 1} / {Config.NUM_REPEATS} ---")
        rng = np.random.default_rng(seed=Config.SEED + repeat_idx)
        shuffled_results = results[:]
        rng.shuffle(shuffled_results)

        train_split = shuffled_results[:train_size]
        calibration_split = shuffled_results[train_size:train_size + calibration_size]
        test_split = shuffled_results[train_size + calibration_size:]

        input_train = [r["input"] for r in train_split]
        true_train = [r["true_value"] for r in train_split]
        pred_train = [r["prediction"] for r in train_split]
        probs_train = [r["probs"] for r in train_split]
        idx_train = [r["row_index"] for r in train_split]

        input_calibration = [r["input"] for r in calibration_split]
        true_calibration = [r["true_value"] for r in calibration_split]
        pred_calibration = [r["prediction"] for r in calibration_split]
        probs_calibration = [r["probs"] for r in calibration_split]
        idx_calibration = [r["row_index"] for r in calibration_split]

        input_test = [r["input"] for r in test_split]
        true_test = [r["true_value"] for r in test_split]
        pred_test = [r["prediction"] for r in test_split]
        probs_test = [r["probs"] for r in test_split]
        idx_test = [r["row_index"] for r in test_split]

        if idx_train:
            X_train = np.stack([E[i].astype(np.float32) for i in idx_train], axis=0)
            X_train = l2norm_rows(X_train)
        else:
            X_train = np.empty((0, D), dtype=np.float32)

        if idx_calibration:
            X_cal = np.stack([E[i].astype(np.float32) for i in idx_calibration], axis=0)
            X_cal = l2norm_rows(X_cal)
        else:
            X_cal = np.empty((0, D), dtype=np.float32)

        if idx_test:
            X_test = np.stack([E[i].astype(np.float32) for i in idx_test], axis=0)
            X_test = l2norm_rows(X_test)
        else:
            X_test = np.empty((0, D), dtype=np.float32)

        if dataset_type in Config.TASK_TYPES['regression_tasks']:
            center_train = np.array([r.get("prediction", None) for r in train_split], dtype=float)
            center_calibration = np.array([r.get("prediction", None) for r in calibration_split], dtype=float)
            center_test = np.array([r.get("prediction", None) for r in test_split], dtype=float)
            if (
                np.any(np.isnan(center_train))
                or np.any(np.isnan(center_calibration))
                or np.any(np.isnan(center_test))
            ):
                raise ValueError("Missing numeric 'prediction' in results. Save it during inference or parse it here.")
            m_train = np.asarray(center_train, dtype=np.float32)
            m_cal = np.asarray(center_calibration, dtype=np.float32)
            m_test = np.asarray(center_test, dtype=np.float32)
            print("n_train:", len(X_train), "n_cal:", len(X_cal), "n_test:", len(X_test))
            r_cal = np.abs(np.asarray(true_calibration, dtype=np.float32) - m_cal)
            print("r_cal: mean,median,std,min,max", r_cal.mean(), np.median(r_cal), r_cal.std(), r_cal.min(), r_cal.max())
            y_train = np.asarray(true_train, dtype=np.float32)
            y_cal = np.asarray(true_calibration, dtype=np.float32)
            y_test = np.asarray(true_test, dtype=np.float32)

        if dataset_type in Config.TASK_TYPES['ordinal_classification'] or dataset_type in Config.TASK_TYPES['multiclass_classification']:
            true_train, pred_train, probs_train = get_prediction_touples(true_train, pred_train, probs_train, dataset_type)
            true_calibration, pred_calibration, probs_calibration = get_prediction_touples(true_calibration, pred_calibration, probs_calibration, dataset_type)
            true_test, pred_test, probs_test = get_prediction_touples(true_test, pred_test, probs_test, dataset_type)

        for mode in modes_to_run:
            if is_multiclass_task:
                Config.MULTICLASS_CP_MODE = mode
                Config.update_paths()
                Config.TASK_TYPE = "multiclass_classification"
                print(f"\n=== Multiclass CP mode: {mode} ===")
            elif is_ordinal_task:
                Config.ORDINAL_CP_MODE = mode
                Config.update_paths()
                Config.TASK_TYPE = "ordinal_classification"
                print(f"\n=== Ordinal CP mode: {mode} ===")
            elif is_regression_task:
                Config.REGRESSION_CP_MODE = mode
                Config.update_paths()
                Config.TASK_TYPE = "regression_tasks"
                print(f"\n=== Regression CP mode: {mode} ===")

            #for alpha in Config.CP_ALPHA:            
            start_time = time.time()
            if is_regression_task:
                if (mode == "quantilized_regression"):
                        # minimal tau set from CP_ALPHA (no retraining per alpha)
                    taus_needed = sorted({
                        round(a/2.0, 4) for a in Config.CP_ALPHA
                    } | {
                        round(1.0 - a/2.0, 4) for a in Config.CP_ALPHA
                    })
                    print(f"[DEBUG] Taus needed for {len(Config.CP_ALPHA)} alphas: {len(taus_needed)} quantiles = {taus_needed}")

                    pred_train_numeric = np.asarray(pred_train, dtype=np.float32)
                    pred_cal_numeric = np.asarray(pred_calibration, dtype=np.float32)
                    pred_test_numeric = np.asarray(pred_test, dtype=np.float32)

                    X_train_aug = _sanitize_features(
                        _augment_with_metadata(X_train, pred_train_numeric, center_train), "train"
                    )
                    X_cal_aug = _sanitize_features(
                        _augment_with_metadata(X_cal, pred_cal_numeric, center_calibration), "calibration"
                    )
                    X_test_aug = _sanitize_features(
                        _augment_with_metadata(X_test, pred_test_numeric, center_test), "test"
                    )

                    print(
                        f"[DEBUG] Feature shapes after augmentation: train={X_train_aug.shape}, cal={X_cal_aug.shape}, test={X_test_aug.shape}"
                    )

                    # cache path for the quantile regressor (per model+dataset+taus)
                    qr_cache_dir = os.path.join(getattr(Config, "ARTIFACTS_DIR", "."), "qr_models")
                    os.makedirs(qr_cache_dir, exist_ok=True)
                    qr_cache_name = (
                        f"qr_{Config.MODEL_NAME_OR_PATH.replace('/','_')}_{dataset_type.replace(',','_')}_{Config.CQR_CACHE_VERSION}.joblib"
                    )
                    qr_cache_path = os.path.join(qr_cache_dir, qr_cache_name)

                    # load or train once
                    if os.path.exists(qr_cache_path):
                        print(f"[DEBUG] Loading cached quantile regressor: {qr_cache_path}")
                        qr_load_start = time.time()
                        qr = load(qr_cache_path)
                        print(f"[DEBUG] Cache loaded in {time.time() - qr_load_start:.2f}s")
                    else:
                        print(f"[DEBUG] Training quantile regressor with {len(taus_needed)} quantiles on {len(X_train)} samples...")
                        print(
                            f"[DEBUG] ⚙️  Hyperparameter grid size={len(Config.CQR_PARAM_GRID)} | CV folds={Config.CQR_CV_FOLDS}"
                        )
                        qr_train_start = time.time()
                        qr = QuantileRegressor(
                            quantiles=taus_needed,
                            param_grid=Config.CQR_PARAM_GRID,
                            cv_folds=Config.CQR_CV_FOLDS,
                            use_pca=Config.CQR_USE_PCA,
                            pca_variance=Config.CQR_PCA_VARIANCE,
                            tail_weight=Config.CQR_PINBALL_TAIL_WEIGHT,
                            early_stopping=Config.CQR_EARLY_STOPPING,
                            early_stopping_rounds=Config.CQR_EARLY_STOPPING_ROUNDS,
                            early_stopping_tol=Config.CQR_EARLY_STOPPING_TOL,
                            validation_fraction=Config.CQR_VALIDATION_FRACTION,
                        )
                        qr.fit(X_train_aug, y_train)
                        print(f"[DEBUG] Training completed in {time.time() - qr_train_start:.2f}s")
                        print(f"[DEBUG] Saving to cache: {qr_cache_path}")
                        dump(qr, qr_cache_path, compress=3)
                        print(f"[DEBUG] Cache saved")
                    # precompute cal/test predictions once for the whole tau set (OPTIMIZED: batch prediction)
                    print(f"[DEBUG] Making batch predictions on calibration set ({len(X_cal_aug)} samples)...")
                    pred_cal_start = time.time()
                    preds_cal_all = qr.predict_quantiles(X_cal_aug, quantiles=taus_needed)   # shape (n_cal, len(taus_needed))
                    print(f"[DEBUG] Cal predictions done in {time.time() - pred_cal_start:.2f}s: shape={preds_cal_all.shape}")
                    
                    print(f"[DEBUG] Making batch predictions on test set ({len(X_test_aug)} samples)...")
                    pred_test_start = time.time()
                    preds_test_all = qr.predict_quantiles(X_test_aug, quantiles=taus_needed)  # shape (n_test, len(taus_needed))
                    print(f"[DEBUG] Test predictions done in {time.time() - pred_test_start:.2f}s: shape={preds_test_all.shape}")
                    
                    # Convert to dict format {tau: array}
                    cal_preds_all = {tau: preds_cal_all[:, i] for i, tau in enumerate(sorted(taus_needed))}
                    test_preds_all = {tau: preds_test_all[:, i] for i, tau in enumerate(sorted(taus_needed))}

                    # CQR calibrator
                    cqr = ConformalizedQuantileRegressionPredictor(asymmetric=True)

                    for alpha in Config.CP_ALPHA:
                        print(f"\n[DEBUG] Processing alpha={alpha:.2f} ({1-alpha:.1%} confidence)...")
                        alpha_start = time.time()
                        
                        q_lo = float(round(alpha/2.0, 4))
                        q_hi = float(round(1.0 - alpha/2.0, 4))

                        # pick or interpolate the base quantiles
                        if q_lo in cal_preds_all and q_hi in cal_preds_all:
                            lower_cal, upper_cal = cal_preds_all[q_lo], cal_preds_all[q_hi]
                            lower_test, upper_test = test_preds_all[q_lo], test_preds_all[q_hi]
                        else:
                            print(f"[DEBUG]   Interpolating q_lo={q_lo}, q_hi={q_hi}")
                            lower_cal  = _interp(cal_preds_all,  q_lo, taus_needed)
                            upper_cal  = _interp(cal_preds_all,  q_hi, taus_needed)
                            lower_test = _interp(test_preds_all, q_lo, taus_needed)
                            upper_test = _interp(test_preds_all, q_hi, taus_needed)

                        probs_cal  = {"lower": lower_cal,  "upper": upper_cal}
                        probs_test = {"lower": lower_test, "upper": upper_test}

                        # α-specific conformal calibration
                        print(f"[DEBUG]   Calibrating CQR...")
                        Q_pair = cqr.fit(y_true=y_cal, y_pred=None, probs_calibration=probs_cal, alpha=alpha)

                        # intervals on test
                        print(f"[DEBUG]   Predicting on test set...")
                        lower, upper = cqr.predict(y_pred=None, probs_test=probs_test, quantiles=Q_pair)

                        coverage = float(np.mean((y_test >= lower) & (y_test <= upper)))
                        avg_size = float(np.mean(upper - lower))
                        std_size = float(np.std(upper - lower))
                        min_size = float(np.min(upper - lower))
                        max_size = float(np.max(upper - lower))
                        print(f"[DEBUG]   ✓ α={alpha:.2f} → coverage={coverage:.3f}, size={avg_size:.3f}±{std_size:.3f} [{min_size:.3f}, {max_size:.3f}] ({time.time()-alpha_start:.2f}s)")
                        print(f"[DEBUG]      🔹 CQR [ADAPTIVE]: Intervals vary by sample (std={std_size:.3f})")

                        conformal_results = ((lower, upper), coverage, avg_size, y_test)
                        # probs_test = base quantiles; keep if your saver expects them
                        print(f"[DEBUG]   Saving results...")
                        #save_cp_results(dataset_type, input_test, y_test, pred_test, probs_test, conformal_results, alpha)
                        total_time = time.time() - start_time
                        record = build_cp_result_record(
                            dataset_type,
                            true_test,
                            pred_test,
                            probs_test,
                            conformal_results,
                            alpha,
                            repeat_idx,
                            mode=mode,
                            seed=Config.SEED + repeat_idx,
                            predictor=cqr,
                            timestamp=total_time
                        )
                        result_path = Config.CONFORMAL_RESULTS_FILE
                        aggregated_results_by_path[result_path].append(record)
                elif mode == "regression":
                    for alpha in Config.CP_ALPHA:
                        baseline_cp = get_predictor(Config.TASK_TYPE, alpha)
                        alpha_start = time.time()
                        q_hat = baseline_cp.fit(true_calibration, pred_calibration, probs_calibration, alpha)
                        conformal_results = baseline_cp.get_conformal_results(true_test, pred_test, probs_test, q_hat)
                        
                        # Extract interval info for standard CP
                        lower, upper = conformal_results[0]
                        coverage = conformal_results[1]
                        avg_size = conformal_results[2]
                        std_size = float(np.std(upper - lower)) if hasattr(upper, '__len__') else 0.0
                        
                        print(f"[DEBUG] [{Config.TASK_TYPE}] α={alpha:.2f} → coverage={coverage:.3f}, size={avg_size:.3f}±{std_size:.3f} ({time.time()-alpha_start:.2f}s)")
                        print(f"[DEBUG]      🔹 Standard [CONSTANT]: All samples get same fixed interval")
                        total_time = time.time() - start_time
                        record = build_cp_result_record(
                            dataset_type,
                            true_test,
                            pred_test,
                            probs_test,
                            conformal_results,
                            alpha,
                            repeat_idx,
                            mode=mode,
                            seed=Config.SEED + repeat_idx,
                            predictor=baseline_cp,
                            timestamp=total_time
                        )
                        result_path = Config.CONFORMAL_RESULTS_FILE
                        aggregated_results_by_path[result_path].append(record)
                else:
                    raise ValueError(f"Unknown regression CP mode: {mode}")
            else: #if it is not regression_task
                for alpha in Config.CP_ALPHA:
                    baseline_cp = get_predictor(Config.TASK_TYPE, alpha)
                    q_hat = baseline_cp.fit(true_calibration, pred_calibration, probs_calibration, alpha)
                    conformal_results = baseline_cp.get_conformal_results(true_test, pred_test, probs_test, q_hat)

                    if hasattr(baseline_cp, "_empty_prob_calibration") and hasattr(baseline_cp, "_empty_prob_test"):
                        empty_cal = getattr(baseline_cp, "_empty_prob_calibration", 0)
                        empty_test = getattr(baseline_cp, "_empty_prob_test", 0)
                        if empty_cal or empty_test:
                            print(
                                f"Warning: skipped {empty_cal} calibration and {empty_test} test samples with empty probability vectors."
                            )

                    print(
                        f" Task: {Config.TASK_TYPE} Confidence: {1-alpha:.2f} Coverage: {conformal_results[1]:.3f}  Size: {conformal_results[2]:.2f}"
                    )
                    if getattr(baseline_cp, "tuned_tau", None) is not None:
                        print(
                            f"  -> Hybrid tau tuned to {baseline_cp.tuned_tau:.3f} (coverage target >= global, size <= mondrian on inner split)"
                        )
                    total_time = time.time() - start_time
                    record = build_cp_result_record(
                        dataset_type,
                        true_test,
                        pred_test,
                        probs_test,
                        conformal_results,
                        alpha,
                        repeat_idx,
                        mode=mode,
                        seed=Config.SEED + repeat_idx,
                        predictor=baseline_cp,
                        timestamp=total_time
                    )
                    result_path = Config.CONFORMAL_RESULTS_FILE
                    aggregated_results_by_path[result_path].append(record)

    for path, records in aggregated_results_by_path.items():
        save_cp_results(path, records)

    if is_multiclass_task:
        Config.MULTICLASS_CP_MODE = original_multiclass_mode
    if is_ordinal_task:
        Config.ORDINAL_CP_MODE = original_ordinal_mode
    if is_regression_task:
        Config.REGRESSION_CP_MODE = original_regression_mode
    Config.update_paths()

if __name__ == "__main__":
    analysis = False # False if you want to run inference and conformal prediction 
    for model_name in Config.BASELINE_MODEL_NAMES:
        for dataset_name in Config.BASELINE_DATASETS:
            if not analysis:
                #if model_name includes llama
                if "llama" in model_name:
                    Config.USE_LLAMA = True
                else:
                    Config.USE_LLAMA = False

                Config.update_model_and_dataset(model_name, dataset_name)
                
                #1: Get model responses
                run_inference()
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
