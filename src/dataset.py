from pyexpat import model
import config
from datasets import load_dataset
from transformers import PegasusTokenizer

class Dataset:

    def __init__(self):
        self.model_ckpt = config.MODEL_CKPT
        self.tokenizer = PegasusTokenizer.from_pretrained(self.model_ckpt)

    def load_data(self):
        train_data = load_dataset(config.DATASET_ID, split="validation")
        return train_data

    def model_inp_gen(self, data):
        model_inputs = self.tokenizer(
            data["article"], max_length=config.MAX_INP_LENGTH, truncation=True)
        with self.tokenizer.as_target_tokenizer():
            labels = self.tokenizer(
                    data["abstract"], max_length=config.MAX_TARGET_LENGTH, truncation=True
                )
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    def run(self):
        data_train = self.load_data()
        train_model_inps = data_train.map(self.model_inp_gen, batched=True)
        return train_model_inps