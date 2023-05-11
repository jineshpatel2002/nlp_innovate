import pickle
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from preprocess import *

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        input_text = request.form['input_text']
        if input_text != '':
            encoder = pickle.load(open('encoder.pkl', 'rb'))
            cv = pickle.load(open('CountVectorizer.pkl', 'rb'))
            model = tf.keras.models.load_model('my_model.h5')
            input_text = preprocess(input_text)
            array = cv.transform([input_text]).toarray()
            pred = model.predict(array)
            a = np.argmax(pred, axis=1)
            prediction = encoder.inverse_transform(a)[0]
        else:
            prediction = "The emotion of this text is..."
        return render_template('index1.html', prediction=prediction)
    else:
        return render_template('index1.html', prediction="The emotion of this text is...")

if __name__ == '__main__':
    app.run(debug=True)
