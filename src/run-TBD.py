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
from conformalprediction.quantileregression import QuantileRegressor
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
from conformalprediction.conformalizedquantileregression import ConformalizedQuantileRegressionPredictor

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
    
    def _resolve_max_memory(dev):
        explicit = getattr(Config, "MAX_MEMORY", None)
        if explicit:
            # ensure keys/values are strings or ints as transformers expects
            parsed = {}
            for key, value in explicit.items():
                if isinstance(value, (int, float)):
                    parsed[key] = f"{value}GiB"
                else:
                    parsed[key] = str(value)
            return parsed

        if not torch.cuda.is_available():
            return None

        util = getattr(Config, "GPU_MEMORY_UTILIZATION", None)
        if util is None:
            return None

        dev_index = dev.index if dev.index is not None else 0
        props = torch.cuda.get_device_properties(dev_index)
        total_gib = props.total_memory / (1024 ** 3)
        alloc_gib = max(int(total_gib * util), 1)

        max_mem = {dev_index: f"{alloc_gib}GiB"}
        cpu_budget = getattr(Config, "CPU_OFFLOAD_GB", None)
        if cpu_budget:
            max_mem["cpu"] = f"{int(cpu_budget)}GiB"
        return max_mem

    model_config = AutoConfig.from_pretrained(Config.MODEL_NAME_OR_PATH)
    device_map = getattr(Config, "DEVICE_MAP", "auto")
    max_memory = _resolve_max_memory(device)
    common_model_kwargs = {
        "torch_dtype": load_type,
        "config": model_config,
        "low_cpu_mem_usage": True
    }
    if device_map:
        common_model_kwargs["device_map"] = device_map
    if max_memory and device_map:
        common_model_kwargs["max_memory"] = max_memory
        if "cpu" in max_memory:
            offload_folder = getattr(Config, "OFFLOAD_FOLDER", None)
            if offload_folder:
                os.makedirs(offload_folder, exist_ok=True)
                common_model_kwargs["offload_folder"] = offload_folder

    # Load model (causal vs seq2seq)
    model_type = model_config.__class__.__name__
    if model_type == "T5Config":
        from transformers import T5ForConditionalGeneration
        model = T5ForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            **common_model_kwargs
        )
        is_seq2seq = True
    elif model_type == "BartConfig":
        from transformers import BartForConditionalGeneration
        model = BartForConditionalGeneration.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            **common_model_kwargs
        )
        is_seq2seq = True
    else:
        model = AutoModelForCausalLM.from_pretrained(
            Config.MODEL_NAME_OR_PATH,
            **common_model_kwargs
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
    from joblib import dump, load  # used for caching the quantile regressor
    import time
    
    print("\n" + "="*80)
    print("[DEBUG] Starting run_conformal_prediction()")
    print("="*80)
    start_time = time.time()

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
        extras.append(preds_arr)

        if centers is not None:
            centers = np.asarray(centers, dtype=np.float32)
            if not np.all(np.isnan(centers)):
                if np.any(np.isnan(centers)):
                    fill = float(np.nanmean(centers))
                    centers = np.where(np.isnan(centers), fill, centers)
                extras.append(centers.reshape(-1, 1))

        if extras:
            extra_cols = np.concatenate(extras, axis=1)
            return np.concatenate([X, extra_cols], axis=1)
        return X

    dataset_type = Config.DS_TYPE

    # Load & filter results
    with open(Config.RESULTS_FILE, 'r', encoding="utf-8") as read_f:
        all_rows = [json.loads(line) for line in read_f]
    results = [r for r in all_rows if r["ds_type"] == dataset_type]
    
    print(f"[DEBUG] Loaded {len(all_rows)} total results, filtered to {len(results)} for dataset_type={dataset_type}")

    # Clean/shuffle/split
    print("[DEBUG] Starting data cleaning...")
    results = cleaning_results(results, dataset_type)
    print(f"[DEBUG] After cleaning: {len(results)} valid results")
    
    random.shuffle(results)

    train_size = int(len(results) * Config.TRAIN_SET_SIZE)
    test_size = int(len(results) * Config.TEST_SET_SIZE)
    calibration_size = int(len(results) * Config.CALIBRATION_SET_SIZE)
    
    print(f"[DEBUG] Data split: train={train_size}, cal={calibration_size}, test={test_size}")

    # Load embeddings (npy or raw)
    print("[DEBUG] Loading embeddings...")
    E, N, D = load_embeddings_any(Config.HIDDEN_OUT, results, dtype=np.float16)
    print(f"[DEBUG] Embeddings loaded: shape=({N}, {D}), dtype={E.dtype}")

    # ---- split lists ----
    true_train = [r["true_value"] for r in results[:train_size]]
    pred_train = [r["prediction"] for r in results[:train_size]]
    probs_train = [r["probs"] for r in results[:train_size]]
    idx_train = [r["row_index"] for r in results[:train_size]]

    true_calibration = [r["true_value"] for r in results[train_size:train_size + calibration_size]]
    pred_calibration = [r["prediction"] for r in results[train_size:train_size + calibration_size]]
    probs_calibration = [r["probs"] for r in results[train_size:train_size + calibration_size]]
    idx_calibration = [r["row_index"] for r in results[train_size:train_size + calibration_size]]

    true_test = [r["true_value"] for r in results[train_size + calibration_size:train_size + calibration_size + test_size]]
    pred_test = [r["prediction"] for r in results[train_size + calibration_size:train_size + calibration_size + test_size]]
    probs_test = [r["probs"] for r in results[train_size + calibration_size:train_size + calibration_size + test_size]]
    idx_test = [r["row_index"] for r in results[train_size + calibration_size:train_size + calibration_size + test_size]]

    # inputs (for saving)
    input_train = [r["input"] for r in results[:train_size]]
    input_calibration = [r["input"] for r in results[train_size:train_size + calibration_size]]
    input_test = [r["input"] for r in results[train_size + calibration_size:train_size + calibration_size + test_size]]

    # ---- pull embeddings ----
    print("[DEBUG] Extracting embeddings for train/cal/test splits...")
    X_train = np.stack([E[i].astype(np.float32) for i in idx_train], axis=0)
    X_cal   = np.stack([E[i].astype(np.float32) for i in idx_calibration], axis=0)
    X_test  = np.stack([E[i].astype(np.float32) for i in idx_test], axis=0)
    print(f"[DEBUG] Embeddings extracted: X_train={X_train.shape}, X_cal={X_cal.shape}, X_test={X_test.shape}")

    # normalize
    print("[DEBUG] L2-normalizing embeddings...")
    X_train = l2norm_rows(X_train)
    X_cal   = l2norm_rows(X_cal)
    X_test  = l2norm_rows(X_test)
    print("[DEBUG] Embeddings normalized")

    print(f"[DEBUG] Final data: n_train={len(X_train)}, n_cal={len(X_cal)}, n_test={len(X_test)}")

    # ---- targets / auxiliaries ----
    y_train = np.asarray(true_train, dtype=np.float32)
    y_cal   = np.asarray(true_calibration, dtype=np.float32)
    y_test  = np.asarray(true_test, dtype=np.float32)

    # optional: “valence” numeric center if you use it elsewhere
    center_train = np.array([r.get("valence", None) for r in results[:train_size]], dtype=float)
    center_cal   = np.array([r.get("valence", None) for r in results[train_size:train_size + calibration_size]], dtype=float)
    center_test  = np.array([r.get("valence", None) for r in results[train_size + calibration_size:train_size + calibration_size + test_size]], dtype=float)
    if np.any(np.isnan(center_cal)) or np.any(np.isnan(center_test)):
        print("Note: 'valence' missing for some rows; only relevant if you use it later.")

    # classification branches (unchanged)
    if dataset_type in Config.TASK_TYPES['ordinal_classification'] or dataset_type in Config.TASK_TYPES['multiclass_classification']:
        tuples_train = get_prediction_touples(true_train, pred_train, probs_train, dataset_type)
        true_train, pred_train, probs_train = tuples_train
        tuples_calibration = get_prediction_touples(true_calibration, pred_calibration, probs_calibration, dataset_type)
        true_calibration, pred_calibration, probs_calibration = tuples_calibration
        tuples_test = get_prediction_touples(true_test, pred_test, probs_test, dataset_type)
        true_test, pred_test, probs_test = tuples_test

    # --- Conformal prediction across task types ---
    for ttype in Config.TASK_TYPES:
        if Config.DS_TYPE not in Config.TASK_TYPES[ttype]:
            continue

        print(f"\n[DEBUG] Processing task type: {ttype}")
        baseline_cp = get_predictor(ttype, None)  # your factory; alpha handled inside loops below

        if ttype in ("weighted_regression", "local_regression"):
            # TODO: your local/clustered CP code
            print(f"[{ttype}] not implemented in this snippet.")
            continue

        elif ttype == "quantilized_regression":
            # ===== CQR PATH =====

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

            X_train_aug = _augment_with_metadata(X_train, pred_train_numeric, center_train)
            X_cal_aug = _augment_with_metadata(X_cal, pred_cal_numeric, center_cal)
            X_test_aug = _augment_with_metadata(X_test, pred_test_numeric, center_test)

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
                save_cp_results(dataset_type, input_test, y_test, pred_test, probs_test, conformal_results, alpha)

        else:
            print(f"\n[DEBUG] Processing standard {ttype}...")
            for alpha in Config.CP_ALPHA:
                alpha_start = time.time()
                q_hat = baseline_cp.fit(true_calibration, pred_calibration, probs_calibration, alpha)
                conformal_results = baseline_cp.get_conformal_results(true_test, pred_test, probs_test, q_hat)
                
                # Extract interval info for standard CP
                lower, upper = conformal_results[0]
                coverage = conformal_results[1]
                avg_size = conformal_results[2]
                std_size = float(np.std(upper - lower)) if hasattr(upper, '__len__') else 0.0
                
                print(f"[DEBUG] [{ttype}] α={alpha:.2f} → coverage={coverage:.3f}, size={avg_size:.3f}±{std_size:.3f} ({time.time()-alpha_start:.2f}s)")
                print(f"[DEBUG]      🔹 Standard [CONSTANT]: All samples get same fixed interval")
                save_cp_results(dataset_type, input_test, true_test, pred_test, probs_test, conformal_results, alpha)
    
    # Final summary
    total_time = time.time() - start_time
    print(f"\n{'='*80}")
    print(f"[DEBUG] Conformal prediction completed in {total_time/60:.1f} minutes ({total_time:.0f}s)")
    print(f"{'='*80}")
    print(f"\n📊 KEY DIFFERENCE BETWEEN METHODS:\n")
    print(f"  🔹 CQR (Quantile Regression)   [ADAPTIVE]")
    print(f"     • Learns separate quantile models (tail-aware τ grid)")
    print(f"     • Interval size VARIES by sample (non-zero std)")
    print(f"     • Coverage adapts to local uncertainty; quality follows quantile fit")
    print(f"     • ✅ Preferred when adaptive intervals are required\n")
    print(f"  🔹 Standard Regression         [CONSTANT]")
    print(f"     • Computes single residual quantile for all samples")
    print(f"     • All samples get SAME fixed interval (std≈0)")
    print(f"     • One-size-fits-all approach")
    print(f"     • Useful baseline for sanity checks")
    print(f"     • 📖 Use for benchmarking\n")
    print(f"{'='*80}\n")
    
    # Save execution summary
    execution_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_type": Config.DS_TYPE,
        "model": Config.MODEL_NAME_OR_PATH,
        "total_execution_time_seconds": total_time,
        "total_execution_time_minutes": total_time / 60,
        "n_alphas": len(Config.CP_ALPHA),
        "n_train": len(X_train),
        "n_cal": len(X_cal),
        "n_test": len(X_test),
        "results_file": Config.CONFORMAL_RESULTS_FILE,
        "log_dir": "results/qr_training_logs",
        "method_comparison": {
            "CQR": "ADAPTIVE - interval size varies by sample (recommended)",
            "Standard": "CONSTANT - all samples get same interval (baseline only)"
        }
    }
    
    log_dir = "results/execution_logs"
    os.makedirs(log_dir, exist_ok=True)
    exec_log_path = os.path.join(log_dir, f"execution_{time.strftime('%Y%m%d_%H%M%S')}.json")
    with open(exec_log_path, 'w') as f:
        json.dump(execution_log, f, indent=2)
    print(f"[DEBUG] Execution log saved: {exec_log_path}")

if __name__ == "__main__":
    analysis = False # False if you want to run inference and conformal prediction
    model_names = [
        #"lzw1008/Emollama-7b",
        #"lzw1008/Emollama-chat-7b",
        #"lzw1008/Emollama-chat-13b",
        #"lzw1008/Emoopt-13b",
        "lzw1008/Emobloom-7b"
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
            run_analysis(model_name, dataset_name)
    
    # Run integrated analysis after all individual analyses are complete
    #print("\nRunning integrated analysis...")
    #run_integrated_analysis()
    #print("Integrated analysis completed!")
