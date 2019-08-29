from flask import Flask, request, jsonify
app = Flask(__name__)

from sklearn.externals import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
lr = joblib.load('log_reg_model.pkl')
Vectorize = joblib.load('vectorize.pkl')


#POST request

@app.route('/is_toxic', methods=['POST'])
def is_toxic():


    data = request.get_json()
    #data = str(data)
    data = Vectorize.transform(data)

    verdict = lr.predict(data)[0]

    if verdict == 1:
        verdict = 'Toxic'
    else:
        verdict = 'Ok'

    response = {'This comment is:': verdict}

    return jsonify(response)




