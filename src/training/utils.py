import evaluate
import numpy as np


def get_sacrebleu_metric_fn(tokenizer):
    metric = evaluate.load('sacrebleu')

    def compute_metric(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]

        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        decoded_preds = [pred.strip() for pred in decoded_preds]
        decoded_labels = [[label.strip()] for label in decoded_labels]
        result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        return {'bleu': result['score']}

    return compute_metric
