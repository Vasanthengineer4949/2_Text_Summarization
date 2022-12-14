DATASET_ID = "ccdv/arxiv-summarization"
MODEL_CKPT = "google/pegasus-large"
MAX_INP_LENGTH = 1024
MAX_TARGET_LENGTH = 256
EVAL_METRIC = "rouge"
BATCH_SIZE = 1
NUM_EPOCHS = 5
LEARNING_RATE = 5.6e-5
WEIGHT_DECAY = 0.01
MODEL_OUT = f"{MODEL_CKPT}-finetuned-arxiv"
INFERENCE = "Vasanth/pegasus-large-finetuned-arxiv"
