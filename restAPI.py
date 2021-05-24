from flask_ngrok import run_with_ngrok
from flask import Flask
from flask import request
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, LongformerForQuestionAnswering, pipeline
import json

from test import qa

app = Flask(__name__)
run_with_ngrok(app)  # starts ngrok when the app is run

# Mantas Pimpe 1813010
# Longformer

@app.route("/")
def detect():
    model_checkpoint = "allenai/longformer-base-4096"
    model = AutoModelForQuestionAnswering.from_pretrained("test-squad-trained")
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    predict = pipeline('question-answering', model=model, tokenizer=tokenizer)

    context = request.args.get('c')
    question = request.args.get('q')
    return json.dumps(qa(predict, context, question, include_score=True))

app.run()
