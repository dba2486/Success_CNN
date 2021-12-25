#run_keras_server.py

from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications import imagenet_utils
from PIL import Image
import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
import cv2
import glob
import matplotlib.pyplot as plt
import os

app = flask.Flask(__name__)
model = None

def load_model():
    model = load_model('jong_s pill_s99.h5')
    
def prepare_image(img):   
    img = cv2.imread(path[0])
    img = cv2.resize(img,dsize=(676,369),interpolation=cv2.INTER_AREA)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    rect = (0,0,700,350)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    rgb = img
    small = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    grad = cv2.morphologyEx(small, cv2.MORPH_GRADIENT, kernel)
    _, bw = cv2.threshold(grad, 0.0, 255.0, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    connected = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)
# using RETR_EXTERNAL instead of RETR_CCOMP
    contours, hierarchy = cv2.findContours(connected.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    mask = np.zeros(bw.shape, dtype=np.uint8)
    for idx in range(len(contours)):
        x, y, w, h = cv2.boundingRect(contours[idx])
        mask[y:y+h, x:x+w] = 0
        cv2.drawContours(mask, contours, idx, (255, 255, 255), -1)
        r = float(cv2.countNonZero(mask[y:y+h, x:x+w])) / (w * h)
        if r > 0.45 and w > 8 and h > 8:
            cv2.rectangle(rgb, (x, y), (x+w-1, y+h-1), (0, 255, 0), 2) 
    edge = cv2.Canny(rgb, 50, 250) 
    resize_edge = cv2.resize(edge,dsize=(68,37),interpolation=cv2.INTER_AREA)
    
    return resize_edge
        
    
@app.route('/predict', methods = ['POST'])
def predict():
    data = {'success':False}
    
    if flask.request.methos =='POST':
        if flask.request.files.get('image'):
            image = glob.glob('C:/Users/kjh97/prac/testset/*.png')
            image = flask.request.files['image'].read()
            image = Image.open(io.BytesIO(image))
            
            
        image = prepare_image(image)
        preds = model.predict(image)
        results = imagenet_utils.decode_predictions(preds)
        data['predictions']=[]

        for(imagenetID, label, prob) in results[0]:
            r = {'rabel':label, 'probability': float(prob)}
            data['predictions'].append(r)
        
        data['success']=True
    return flask.jsonify(data)

if__name__ == '__main__':
    print(('*loading keras model and flask starting server...'
           'please wait until server has fully started'))
    load_model()
    app.run()
    
    