# -*- coding: utf-8 -*-
"""
Face models required for age-gender estimation
"""


import os
import sys
import cv2
import numpy as np
import mxnet as mx

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'common'))
import face_preprocess

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'mtcnn'))
from mtcnn_detector import MtcnnDetector


def load_model(model_str, image_size, layer, ctx=mx.cpu()):
  _vec = model_str.split(',')
  assert len(_vec)==2
  prefix = _vec[0]
  epoch = int(_vec[1])
  print('loading',prefix, epoch)
  sym, arg_params, aux_params = mx.model.load_checkpoint(prefix, epoch)
  all_layers = sym.get_internals()
  sym = all_layers[layer+'_output']
  model = mx.mod.Module(symbol=sym,data_names=('data','stage_num0','stage_num1','stage_num2'),context=ctx, label_names = None)
  model.bind(data_shapes=[('data', (1, 3, image_size[0], image_size[1])),('stage_num0',(1,3)),('stage_num1',(1,3)),('stage_num2',(1,3))])
  model.set_params(arg_params, aux_params)
  return model

  
class FaceModel:
    def __init__(self, age_model_str, gender_model_str, detection_model_path, image_size=(64,64), det=0):
        self.age_model_str = age_model_str 
        self.gender_model_str = gender_model_str
        self.image_size = image_size
        self.det = det
        self.mtcnn_path = detection_model_path
        
        if mx.context.num_gpus()>0:
          ctx = mx.gpu(0)
        else:
          ctx = mx.cpu()

        self.age_model = None
        self.gender_model = None
        
        if len(self.age_model_str)>0:
          self.age_model = load_model(self.age_model_str, self.image_size, layer='_mulscalar16', ctx=ctx)
        if len(self.gender_model_str)>0:
          self.gender_model = load_model(self.gender_model_str, self.image_size, layer='_mulscalar16', ctx=ctx)      
         
        self.det_minsize = 50
        self.det_threshold = [0.6,0.7,0.8]
        
        if self.det==0:
          detector = MtcnnDetector(model_folder=self.mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=self.det_threshold)
        else:
          detector = MtcnnDetector(model_folder=self.mtcnn_path, ctx=ctx, num_worker=1, accurate_landmark = True, threshold=[0.0,0.0,0.2])
        self.detector = detector


    def detect_faces(self, img):
        ret = self.detector.detect_face(img)
        return ret

    def align_face(self, img, bbox, points):
        bbox = bbox[0:4]
        landmark = points[:].reshape((2,5)).T
        aligned = face_preprocess.preprocess(img, bbox=bbox, landmark = landmark, image_size='112,112')
        return aligned
    
    def preprocess_input(self, face_img):
        face_img = cv2.resize(face_img, self.image_size)
        nimg = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        aligned = np.transpose(nimg, (2,0,1))
        input_blob = np.expand_dims(aligned, axis=0)
        data = mx.nd.array(input_blob)
        db = mx.io.DataBatch(data=(data,mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]])))
        return db
  
    def predict_age(self, db):
        self.age_model.forward(db, is_train=False)
        output = self.age_model.get_outputs()[0].asnumpy()
        return output
    
    def predict_gender(self, db):
        self.gender_model.forward(db, is_train=False)
        output = self.gender_model.get_outputs()[0].asnumpy()
        return output
    
