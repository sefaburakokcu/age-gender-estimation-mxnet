# -*- coding: utf-8 -*-
"""
API for age/gender estimation
"""

import os
import sys

import cv2
import numpy as np
import streamlit as st 
from PIL import Image
sys.path.append(os.path.join(os.path.dirname(__file__), 'src', 'api'))
import face_model
  

age_model_str =  './models/ssr-net/age_model/model,0'
gender_model_str = './models/ssr-net/gender_model/model,0'

model = face_model.FaceModel(age_model_str, gender_model_str)

    
def convert_to_np(image):
    image = np.array(image)  
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) 
    return image

def convert_to_pil(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image= Image.fromarray(image)
    return image

def predict(image):   
    ret = model.detect_faces(image)
    
    output = []
    if ret is None:
        print('No face detected!')
    else:
        
        bboxes, points = ret
        for i,(bbox,landmarks) in enumerate(zip(bboxes, points)):
            aligned = model.align_face(image, bbox, landmarks)
            db = model.preprocess_input(aligned)
            age = model.predict_age(db)[0][0]
            gender_prob = model.predict_gender(db)[0][0]
            if gender_prob>0.5:
                gender = 0 #'Male'
            else: 
                gender = 1 #'Female'
            age = int(age)
            print('Face: %d, Age: %f, Gender: %s' %(i,age,gender))
            bbox = [int(i) for i in bbox[:-1]]
            output.append((bbox, age, gender))
    return output

def draw_predictions(image, predictions):
    if len(predictions)>0:
        for bbox, age, gender in predictions:
            pt1 = (bbox[0],bbox[1])
            pt2 = (bbox[2],bbox[3])
            if gender == 0:
                gender_str = 'Male'
                color = (255,0,0)
            else:
                gender_str = 'Female'
                color = (0,0,255)

            cv2.rectangle(image, pt1, pt2, color, 2)
            text = 'Age: {}'.format(age)
        
            cv2.putText(image, text, org=pt1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,255,0), thickness=1)
        image = convert_to_pil(image)
        return image
    else:
        return image
    
def main():
    st.title("Age-Gender Estimation Demo")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg","png","jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
    
        st.write("Predicting age and gender for faces...")
        
        image_np = convert_to_np(image)
    
        predictions  = predict(image_np)
        
        if len(predictions) > 0:
            image = draw_predictions(image_np, predictions)
            
        st.image(image, caption='Uploaded Image.', use_column_width=True)
        st.write('')
#        st.write('Predicitions:')
#        st.write(predictions)
    st.markdown('''<html lang="en">
<body>
    
  	<div>
  		<p>Designed by <em><a href="https://github.com/sefaburakokcu/">Sefa Burak OKCU</a></em></p> 
	</div>
</body>
</html>''', unsafe_allow_html=True)

if __name__ == '__main__':
    main()