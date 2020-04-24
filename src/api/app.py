# -*- coding: utf-8 -*-
"""
API for age/gender estimation
"""


import face_model
import numpy as np

from PIL import Image
from flask import Flask, request



age_model_str =  '../../models/ssr-net/age_model/model,0'
gender_model_str = '../../models/ssr-net/gender_model/model,0'

model = face_model.FaceModel(age_model_str, gender_model_str)

app = Flask(__name__)
app.config['SECRET_KEY'] = b'_5#y2L"F4Q8zfrtU5L\n\xec]/'


@app.route('/')
def home():
  return 'Hello, This is an Age-Gender Estimation API built by Sefa Burak OKCU.'


@app.route('/estimate', methods=['POST'])
def estimate():
    try:
        image = Image.open(request.files['data']).convert('RGB')

        if image is None:
          print('image is None')
          return '-1'

        assert not isinstance(image, list)
        image = np.asarray(image)
        ret = model.detect_faces(image)
       
        if ret is None:
            print('No face detected!')
            return '-1'
        else:
            output = []
            bboxes, points = ret
            for i,(bbox,landmarks) in enumerate(zip(bboxes, points)):
                aligned = model.align_face(image, bbox, landmarks)
                db = model.preprocess_input(aligned)
                age = model.predict_age(db)[0][0]
                gender_prob = model.predict_gender(db)[0][0]
                if gender_prob>0.5:
                    gender = 'Male'
                else: 
                    gender = 'Female'
                print('Face: %d, Age: %f, Gender: %s' %(i,age,gender))
                age = str(age)
                bbox = [int(i) for i in bbox]
                output.append((str(bbox), age, gender))
            return {'prediction':output}
    except Exception as ex:
        print(ex)
        return '-2'


if __name__ == '__main__':
    app.run('0.0.0.0', port=8080, debug=False)