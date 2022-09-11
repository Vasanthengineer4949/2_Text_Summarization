DATASET_ID = "ccdv/arxiv-summarization"
MODEL_CKPT = "google/pegasus-large"
MAX_INP_LENGTH = 1024
MAX_TARGET_LENGTH = 256
EVAL_METRIC = "rouge"
BATCH_SIZE = 4
NUM_EPOCHS = 8
MODEL_OUT = f"{MODEL_CKPT}-finetuned-arxiv"
