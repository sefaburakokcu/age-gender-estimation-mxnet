
"""
Test age-gender prediction for faces
"""

import os
import sys
import cv2
import numpy as np
 
sys.path.append(os.path.join(os.path.dirname(__file__), 'api'))
import face_model


age_model_str =  '../models/ssr-net/age_model/model,0'
gender_model_str = '../models/ssr-net/gender_model/model,0'

model = face_model.FaceModel(age_model_str, gender_model_str)
    

def draw_predictions(image, predictions):
    if len(predictions)>0:
        for bbox, age, gender in predictions:
            print(bbox[-1])
            if bbox[-1] > 0.9: # score threshold
                pt1 = (int(bbox[0]),int(bbox[1]))
                pt2 = (int(bbox[2]),int(bbox[3]))
                if gender == 'Male':
                    color = (255,0,0)
                else:
                    color = (0,0,255)
    
                cv2.rectangle(image, pt1, pt2, color, 2)
                text = 'Age: {}'.format(age)
            
                cv2.putText(image, text, org=pt1, fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.4, color=(0,255,0), thickness=1)
        return image
    else:
        return image
    
def predict(image):
    ret = model.detect_faces(image)
    if ret is not None:
        predictions = []
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
            predictions.append((bbox,int(age),gender))
        return predictions
    return ret
            
    
def main():
    image = cv2.imread('../../data/faces.png')
    predictions = predict(image)
    
    image = draw_predictions(image, predictions)
    cv2.imwrite("output_image.jpg", image)
    cv2.imshow("Output Image", image)   
    cv2.waitKey(0)
    

if __name__ == '__main__':
    main()



