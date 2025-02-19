class Config:
    # Model settings
    MODEL_NAME_OR_PATH = "lzw1008/Emollama-chat-7b"
    USE_LORA = False
    USE_LLAMA = True
    LOAD_TYPE = "float16"  # or "float32"
    
    # Data paths
    INFER_FILE = "data/AEB.json"
    PREDICT_FILE = "predicts/predict.json"
    
    # Generation settings
    BATCH_SIZE = 16
    SEED = 123
    MAX_NEW_TOKENS = 256
    
    # Generation config
    GENERATION_CONFIG = {
        "temperature": 0.9,
        "top_k": 30,
        "top_p": 0.6,
        "do_sample": True,
        "num_beams": 1,
        "repetition_penalty": 1.2,
        "max_new_tokens": MAX_NEW_TOKENS
    }
