import config
import torch
import gc
import numpy as np
import evaluate
from nltk.tokenize import sent_tokenize
from transformers import PegasusTokenizer, AutoModelForSeq2SeqLM
from transformers import Seq2SeqTrainingArguments
from transformers import Seq2SeqTrainer
from transformers import DataCollatorForSeq2Seq

class Model:
    def __init__(self, data, model_ckpt):
        self.data = data
        self.data = self.data.remove_columns(["article", "abstract"])
        self.tokenizer = PegasusTokenizer.from_pretrained(model_ckpt)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_ckpt)
        torch.cuda.empty_cache()
        gc.collect()

    def training_args(self):
        logging_steps = len(self.data) // config.BATCH_SIZE

        args = Seq2SeqTrainingArguments(
            output_dir= config.MODEL_OUT,
            learning_rate= config.LEARNING_RATE,
            per_gpu_train_batch_size=config.BATCH_SIZE,
            weight_decay= config.WEIGHT_DECAY,
            save_total_limit=3,
            num_train_epochs=config.NUM_EPOCHS,
            predict_with_generate=True,
            logging_steps=logging_steps,
            push_to_hub=True,
            fp16=True
        )

        return args
    
    def compute_metric_rouge(self, eval_pred):
        rouge_score = evaluate.load(config.EVAL_METRIC)
        predictions, labels = eval_pred
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        decoded_preds = ["\n".join(sent_tokenize(pred.strip())) for pred in decoded_preds]
        decoded_labels = ["\n".join(sent_tokenize(label.strip())) for label in decoded_labels]
        result = rouge_score.compute(
            predictions=decoded_preds, references=decoded_labels, use_stemmer=True
        )
        result = {key: value*100 for key, value in result.items()}
        return {k: round(v, 4) for k, v in result.items()}

    def train_model(self):
        torch.cuda.empty_cache()
        gc.collect()
        trainer = Seq2SeqTrainer(
                    self.model,
                    self.training_args(),
                    train_dataset=self.data,
                    eval_dataset=self.data,
                    data_collator=DataCollatorForSeq2Seq(self.tokenizer, model=self.model),
                    tokenizer=self.tokenizer,
                    compute_metrics=self.compute_metric_rouge
                )
        torch.cuda.empty_cache()
        gc.collect()
        trainer.train()
        trainer.push_to_hub()