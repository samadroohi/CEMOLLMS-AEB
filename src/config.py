import os


class Config:
    # ---------- Model settings ----------
    MODEL_NAME_OR_PATH = None
    USE_LORA = False
    USE_LLAMA = True
    LOAD_TYPE = "float16"  # or "float32"

    # ---------- Data paths ----------
    INFER_FILE = "data/AEB.json"

    TASK_TYPES = {
        "ordinal_classification": ["EI-oc", "V-oc", "SST5", "TDT"], 
        "weighted_regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank", "SST"],
        "classification": [], 
        
        "regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank","SST"], 
        "local_regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank","SST"], 
        "multiclass_classification": [ "GoEmotions", "E-c"]
    }

    BASELINE_MODEL_NAMES = [
        "lzw1008/Emollama-7b",
        "lzw1008/Emobloom-7b",
        "lzw1008/Emollama-chat-7b",
        "lzw1008/Emollama-chat-13b",
        "lzw1008/Emoopt-13b",
    ]
    BASELINE_DATASETS = [
        "V-oc",
        "EI-oc",
        "SST5",
        "E-c",
        "GoEmotions",
    ]

    VERBOSE = False
    CP_ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5]

    # number of repeat runs for stochastic models
    NUM_REPEATS = 10

    # ---------- Multiclass CP settings ----------
    MULTICLASS_CP_MODE = "hybrid"  # options: "global", "mondrian", "hybrid"
    MULTICLASS_CP_MODES = ["global", "mondrian", "hybrid"]
    MULTICLASS_RARE_SHRINK = 5     # τ parameter for rare-class shrinkage

    # ---------- Ordinal CP settings ----------
    ORDINAL_CP_MODE = "global"     # options mirror multiclass
    ORDINAL_CP_MODES = ["global", "mondrian", "hybrid"]
    ORDINAL_RARE_SHRINK = 5        # τ parameter for rare-class shrinkage
    ORDINAL_HYBRID_TUNE = True     # if True, tune τ automatically
    ORDINAL_HYBRID_TAU_GRID = [
        0.25,
        0.5,
        1.0,
        2.0,
        5.0,
        10.0,
        20.0,
        40.0,
        80.0,
        160.0,
        320.0,
    ]
    ORDINAL_HYBRID_TUNE_HOLDOUT = 0.4  # fraction of calibration points for inner validation
    ORDINAL_HYBRID_MIN_COVERAGE_GAIN = 0.02  # require hybrid coverage >= global + margin
    ORDINAL_HYBRID_SIZE_TOL = 1e-6  # allow small numerical slack when comparing set sizes

    # dataset selection
    DS_TYPE = None
    TRAIN_SET_SIZE = 0.6
    CALIBRATION_SET_SIZE = 0.2
    TEST_SET_SIZE = 0.2

    # ---------- Generation settings ----------
    BATCH_SIZE = 8
    SEED = 123
    MAX_NEW_TOKENS = 256
    TEMPERATURES = [0.7, 0.9, 1.0, 1.5, 2.0]

    # Main (sampling) generation config — unchanged
    GENERATION_CONFIG = {
        "temperature": TEMPERATURES[1],
        "top_k": 30,
        "top_p": 0.6,
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "max_new_tokens": MAX_NEW_TOKENS
    }

    # NEW (optional): a deterministic center you can use when you want stable numeric outputs
    # (useful if later residuals are computed from this center)
    USE_DETERMINISTIC_CENTER = True
    CENTER_GENERATION_CONFIG = {
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "do_sample": False,
        "num_beams": 1,
        "repetition_penalty": 1.0,
        "max_new_tokens": MAX_NEW_TOKENS
    }

    # ---------- Outputs (set by update_paths) ----------
    RESULTS_FILE = None
    CONFORMAL_RESULTS_FILE = None
    PLOTS_DIR = None

    # NEW: where to dump pooled hidden-state vectors (memmap .npy)
    HIDDEN_OUT = None

    # ---------- Extra knobs for saving what CP needs ----------
    # Save the first float parsed from the model's response (e.g., valence in [0,1])
    SAVE_VALENCE = True
    # L2-normalize hidden vectors on write (recommended if you’ll use cosine distance later)
    NORMALIZE_EMB = True
    # Storage dtype for hidden vectors (float16 saves space; switch to float32 if you prefer)
    HIDDEN_DTYPE = "float16"  # or "float32"

    # ---------- Valid data types ----------
    VALID_D_TYPES = {
        "EI-reg": {"min": 0, "max": 1},
        "V-A,V-M,V-NYT,V-T": {"min": -4, "max": 4},
        "SST": {"min": 0, "max": 1},
        "Emobank": {"min": 1, "max": 5},
        "V-reg": {"min": 0, "max": 1},
        "EI-oc": {
            "0": "0: no E can be inferred",
            "1": "1: low amount of E can be inferred",
            "2": "2: moderate amount of E can be inferred",
            "3": "3: high amount of E can be inferred"
        },
        "V-oc": {
            "3": "3: very positive mental state can be inferred",
            "2": "2: moderately positive mental state can be inferred",
            "1": "1: slightly positive mental state can be inferred",
            "0": "0: neutral or mixed mental state can be inferred",
            "-1": "-1: slightly negative mental state can be inferred",
            "-2": "-2: moderately negative mental state can be inferred",
            "-3": "-3: very negative mental state can be inferred"
        },
        "SST5": {
            "0": "0: very negative",
            "1": "1: negative",
            "2": "2: neutral",
            "3": "3: positive",
            "4": "4: very positive"
        },
        "TDT": {"1": "1: positive", "0": "0: neutral", "-1": "-1: negative"},
        "GoEmotions": {
            "0": "neutral", "1": "anger", "2": "disgust", "3": "fear",
            "4": "joy", "5": "sadness", "6": "surprise"
        },
        "E-c": {
            "0": "neutralornoemotion",
            "1": "anger", "2": "anticipation", "3": "disgust", "4": "fear",
            "5": "joy", "6": "love", "7": "optimism", "8": "pessimism",
            "9": "sadness", "10": "surprise", "11": "trust"
        }
    }

    @classmethod
    def update_model_and_dataset(cls, model_name, dataset_type):
        cls.MODEL_NAME_OR_PATH = model_name
        cls.DS_TYPE = dataset_type
        cls.update_paths()

    @classmethod
    def update_paths(cls):
        if cls.MODEL_NAME_OR_PATH is None or cls.DS_TYPE is None:
            return
        model_name_short = cls.MODEL_NAME_OR_PATH.split('/')[-1]
        temperature = cls.GENERATION_CONFIG["temperature"]

        # Existing artifacts
        cls.RESULTS_FILE = f"results/responses/{cls.DS_TYPE}/temp_{temperature}/{model_name_short}.json"
        cls.CONFORMAL_RESULTS_FILE = cls._build_conformal_results_path(model_name_short, temperature)
        cls.PLOTS_DIR = f"results/plots/{cls.DS_TYPE}/temp_{temperature}/{model_name_short}"

        # NEW: hidden-state dump path (aligned row-by-row with RESULTS_FILE)
        cls.HIDDEN_OUT = f"results/hidden/{cls.DS_TYPE}/temp_{temperature}/{model_name_short}.npy"

    @classmethod
    def _build_conformal_results_path(cls, model_name_short: str, temperature: float) -> str:
        """Construct the conformal-results path, adding mode tags for multiclass runs."""
        base_dir = f"results/conformal_results/{cls.DS_TYPE}/temp_{temperature}"
        suffix = ""
        if cls.DS_TYPE in cls.TASK_TYPES.get("multiclass_classification", []):
            suffix = cls._build_mode_suffix(
                getattr(cls, "MULTICLASS_CP_MODE", None),
                getattr(cls, "MULTICLASS_RARE_SHRINK", None),
            )
        elif cls.DS_TYPE in cls.TASK_TYPES.get("ordinal_classification", []):
            suffix = cls._build_mode_suffix(
                getattr(cls, "ORDINAL_CP_MODE", None),
                getattr(cls, "ORDINAL_RARE_SHRINK", None),
            )

        filename = f"{model_name_short}{suffix}.json"
        return os.path.join(base_dir, filename)

    @classmethod
    def get_all_datasets(cls):
        datasets = set()
        for names in cls.TASK_TYPES.values():
            datasets.update(names or [])
        return sorted(datasets)

    @classmethod
    def get_baseline_models(cls):
        return list(getattr(cls, "BASELINE_MODEL_NAMES", []))

    @classmethod
    def get_baseline_datasets(cls):
        datasets = getattr(cls, "BASELINE_DATASETS", None)
        if datasets:
            return list(datasets)
        return cls.get_all_datasets()

    @staticmethod
    def _sanitize_tag(tag: str) -> str:
        tag = tag.lower()
        return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "-" for ch in tag)

    @classmethod
    def _build_mode_suffix(cls, mode, tau) -> str:
        mode_str = str(mode or "").strip().lower()
        if not mode_str:
            return ""

        parts = [mode_str]
        if tau is not None and mode_str in {"hybrid", "mondrian"}:
            try:
                tau_value = float(tau)
            except (TypeError, ValueError):
                tau_value = None
            if tau_value is not None and tau_value > 0:
                if tau_value.is_integer():
                    tau_str = str(int(tau_value))
                else:
                    tau_str = str(tau_value).replace(".", "p")
                parts.append(f"tau{tau_str}")

        safe_tag = cls._sanitize_tag("_".join(parts))
        return f"__{safe_tag}" if safe_tag else ""

    @classmethod
    def get_multiclass_modes(cls):
        """Return the list of multiclass CP modes to evaluate in the current run."""
        modes = getattr(cls, "MULTICLASS_CP_MODES", None)
        normalized = []
        if modes:
            for mode in modes:
                mode_str = str(mode).strip().lower()
                if mode_str:
                    normalized.append(mode_str)
        if normalized:
            unique_modes = []
            for mode in normalized:
                if mode not in unique_modes:
                    unique_modes.append(mode)
            if unique_modes:
                return unique_modes
        default_mode = getattr(cls, "MULTICLASS_CP_MODE", None)
        return [default_mode] if default_mode else ["hybrid"]

    @classmethod
    def get_ordinal_modes(cls):
        """Return the list of ordinal CP modes to evaluate in the current run."""
        modes = getattr(cls, "ORDINAL_CP_MODES", None)
        normalized = []
        if modes:
            for mode in modes:
                mode_str = str(mode).strip().lower()
                if mode_str:
                    normalized.append(mode_str)

        if normalized:
            unique_modes = []
            for mode in normalized:
                if mode not in unique_modes:
                    unique_modes.append(mode)
            if unique_modes:
                return unique_modes

        default_mode = getattr(cls, "ORDINAL_CP_MODE", None)
        return [default_mode] if default_mode else ["global"]
