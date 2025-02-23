class Config:
    # Model settings
    MODEL_NAME_OR_PATH = "lzw1008/Emollama-chat-7b"
    USE_LORA = False
    USE_LLAMA = True
    LOAD_TYPE = "float16"  # or "float32"
    TEMPERATURES = [0.7,0.9, 1.0, 1.5, 2.0]
    # Data paths
    INFER_FILE = "data/AEB.json"
    TASK_TYPES = {
        "classification": ["SST5", "TDT"],
        "ordinal_classification": ["EI-oc", "V-oc"],
        "regression": ["EI-reg", "V-reg", "V-A,V-M,V-NYT,V-T", "SST", "Emobank"],
        "multiclass_classification": [ "GoEmotion", "E-c"]
    }
    VERBOSE = False  # Set to True if you want detailed output
    CP_ALPHA = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    DS_TYPE = "EI-oc"
    CALIBRATION_RATE = 0.1
    # Generation settings
    BATCH_SIZE = 16
    SEED = 123
    MAX_NEW_TOKENS = 256
    
    # Generation config
    GENERATION_CONFIG = {
        "temperature": TEMPERATURES[1],
        "top_k": 30,
        "top_p": 0.6,
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "max_new_tokens": MAX_NEW_TOKENS
    }
    RESULTS_FILE = f"results/responses/{DS_TYPE}/temp_{GENERATION_CONFIG['temperature']}/{MODEL_NAME_OR_PATH.split('/')[-1]}.json"
    CONFORMAL_RESULTS_FILE = f"results/conformal_results/{DS_TYPE}/temp_{GENERATION_CONFIG['temperature']}/{MODEL_NAME_OR_PATH.split('/')[-1]}.json"
    PLOTS_DIR = f"results/plots/{DS_TYPE}/temp_{GENERATION_CONFIG['temperature']}/{MODEL_NAME_OR_PATH.split('/')[-1]}"

    # Valid data types for each task
    VALID_D_TYPES = {
        "EI-reg": {"min": 0, "max": 1},  # Emotion intensity scores from 0 to 1
        "EI-oc": {
            "0": "0: no E can be inferred",
            "1": "1: low amount of E can be inferred",
            "2": "2: moderate amount of E can be inferred",
            "3": "3: high amount of E can be inferred"
            }
        }
    #REVERSE_VALID_D_TYPES = {
     #       v: k for key, sub_dict in VALID_D_TYPES.items() if isinstance(sub_dict, dict) 
      #      for k, v in sub_dict.items()
       # }

