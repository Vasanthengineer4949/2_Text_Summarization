import evaluate
import config
from nltk.tokenize import sent_tokenize

class Metric:
    def __init__(self, data):
        self.data = data
        self.metric = config.EVAL_METRIC

    def three_sentence_summary(self, text):
        return "\n".join(sent_tokenize(text)[:3])

    def evaluate_baseline(self, dataset, metric):
        summaries = [self.three_sentence_summary(text) for text in dataset["article"]]
        return metric.compute(predictions=summaries, references=dataset["abstract"])

    def compute_rouge(self):
        score = self.evaluate_baseline(self.data, evaluate.load("rouge"))
        rouge_names = ["rouge1", "rouge2", "rougeL", "rougeLsum"]
        rouge_dict = dict((rn, round(score[rn]* 100, 2)) for rn in rouge_names)
        return rouge_dict