import torch
from datasets import load_dataset, load_metric
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, LongformerForQuestionAnswering, pipeline
import transformers
from transformers import LongformerModel, TrainingArguments, Trainer
from transformers import default_data_collator
import numpy as np
from tqdm.auto import tqdm
import collections

# Mantas Pimpe 1813010
# Longformer

def qa(predict, context, question, include_score=True, impossible_question_prefix="No answer"):
    p = predict(question=question, context=context, handle_impossible_answer=True, max_seq_len=384)
    score = f" | score: {p['score']}" if include_score else ""
    if (p['start'] == p['end']):
        return impossible_question_prefix + score
    else:
        return f"{p['answer']}" + score

if __name__ == '__main__':
    model_checkpoint = "allenai/longformer-base-4096"
    model = AutoModelForQuestionAnswering.from_pretrained("test-squad-trained")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    predict = pipeline('question-answering', model=model, tokenizer=tokenizer)

    context = r"""
    There is no easy answer to this question due to the many different classifications of computers. The first mechanical computer, created by Charles Babbage in 1822, doesn't resemble what most would consider a computer today. Therefore, this page provides a listing of each of the computer firsts, starting with the Difference Engine and leading up to the computers we use today.
    """

    question = "Who Invented the First Computer?"
    print(qa(predict, context, question, include_score=True))
