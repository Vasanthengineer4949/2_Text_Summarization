import torch
import gc
import config
from dataset import Dataset
from metric import Metric
from model import Model

if __name__ == "__main__":
    dataset = Dataset()
    train_data = dataset.run()
    torch.cuda.empty_cache()
    gc.collect()
    metric = Metric(train_data)
    torch.cuda.empty_cache()
    gc.collect()
    rouge_met = metric.compute_rouge()
    print("Initial Rouge", rouge_met)
    model = Model(train_data, config.MODEL_CKPT)
    model.train_model()


