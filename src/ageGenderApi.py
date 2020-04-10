
"""
This file contains functions required for Age-Gender estimation.
"""


import mxnet as mx


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


def preprocess_image(image):
    image = image[:, :, ::-1]
    image = np.transpose(image,(2,0,1)) 
    input_blob = np.expand_dims(image, axis=0)
    data = mx.nd.array(input_blob)
    db = mx.io.DataBatch(data=(data,mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]]),mx.nd.array([[0,1,2]])))    
    return db

class Age:
    """Class for age estimation."""
    
    def __init__(self, age_model_str):
        self.model = load_model(age_model_str)
        
    def predict_age(self, db):
        self.model.forward(db, is_train=False)
        output = self.model.get_outputs()[0].asnumpy()
        return output
    

class Gender:
    """Class for gender estimation."""
    
    def __init__(self, gender_model_str):
        self.model = load_model(gender_model_str)
    
    def predict_gender(self, db):
        self.model.forward(db, is_train=False)
        output = self.model.get_outputs()[0].asnumpy()
        return output
    
    
    
    
    
    
    
    