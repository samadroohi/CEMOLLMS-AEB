class Config:
    # ---------- Model settings ----------
    MODEL_NAME_OR_PATH = None
    USE_LORA = False
    USE_LLAMA = True
    LOAD_TYPE = "float16"  # or "float32"

    # ---------- Data paths ----------
    INFER_FILE = "data/AEB.json"

    TASK_TYPES = {
        "weighted_regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank", "SST"],
        "classification": [], 
        "ordinal_classification": ["EI-oc", "TDT","V-oc", "SST5"], 
        "regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank","SST"], 
        "local_regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "Emobank","SST"], 
        "multiclass_classification": [ "GoEmotions", "E-c"]
    }

    VERBOSE = False
    CP_ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

    # dataset selection
    DS_TYPE = None
    CALIBRATION_RATE = 0.1

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
        cls.CONFORMAL_RESULTS_FILE = f"results/conformal_results/{cls.DS_TYPE}/temp_{temperature}/{model_name_short}.json"
        cls.PLOTS_DIR = f"results/plots/{cls.DS_TYPE}/temp_{temperature}/{model_name_short}"

        # NEW: hidden-state dump path (aligned row-by-row with RESULTS_FILE)
        cls.HIDDEN_OUT = f"results/hidden/{cls.DS_TYPE}/temp_{temperature}/{model_name_short}.npy"
