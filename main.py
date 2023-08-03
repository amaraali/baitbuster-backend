from flask import Flask, request, jsonify
import re
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertConfig, AutoTokenizer, AutoModel
from tensorflow_addons.optimizers import AdamW
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)


def clean_text(title):
    # Convert to lower case
    title = title.lower()
    # Remove additional white spaces
    title = re.sub('[\s]+', ' ', title)
    # Trim
    title = title.strip('\'"')
    # Clean per words
    words = title.split()
    tokens = []
    for ww in words:
        # Split repeated word
        for w in re.split(r'[-/\s]\s*', ww):
            # Replace two or more with two occurrences
            pattern = re.compile(r"(.)\1{1,}", re.DOTALL)
            w = pattern.sub(r"\1\1", w)
            # Strip punctuation
            w = w.strip('\'"?,.')
            # Check if the word consists of two or more alphabets
            val = re.search(r"^[a-zA-Z][a-zA-Z][a-zA-Z]*$", w)
            # Add tokens
            tokens.append(w.lower())
    title = " ".join(tokens)
    return title


@app.route('/predict', methods=['POST'])
def predict_clickbait():
    title = request.json['title']

    # Cleaning title
    cleaned_title = clean_text(title)
    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained('indobenchmark/indobert-base-p2')
    # Tokenize and preprocess the title
    encoded_title = tokenizer(
        cleaned_title,
        add_special_tokens=True,
        max_length=30,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='tf'
    )

    input_ids = encoded_title['input_ids']
    attention_mask = encoded_title['attention_mask']

    # Load the model
    with tf.keras.utils.custom_object_scope({'TFBertModel': TFBertModel}):
        model = tf.keras.models.load_model('BERTBILSTM_18072023.h5')
    # Make prediction
    prediction = model.predict([input_ids, attention_mask])
    result = 'Clickbait' if prediction[0][0] > 0.5 else 'Non-Clickbait'

    return jsonify({'prediction': result})


if __name__ == '__main__':
    app.run(debug=True)
