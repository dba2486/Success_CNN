from django.http import HttpResponse
from django.shortcuts import render
from tensorflow.keras.models import load_model
import tensorflow as tf
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
import glob
import requests

def home(request):
    return render(request,"home.html")

def result(request):
# mask 
    path = glob.glob('C:/Users/kjh97/lastprac/rtest/*.jpg')
    img1 = path[0]   
    response = requests.post(
        'https://api.remove.bg/v1.0/removebg',
        files={'image_file': open(img1, 'rb')},
        data={'size': 'auto'},
        headers={'X-Api-Key': '5D1ncrBcYV4rx32KCGSku7LQ'},
    )
    if response.status_code == requests.codes.ok:
        with open('C:/Users/kjh97/lastprac/rr/0.jpg', 'wb') as out:
            out.write(response.content)
    else:
        print("Error:", response.status_code, response.text)

    path2 =('C:/Users/kjh97/lastprac/rr/0.jpg')
    img = cv2.imread(path2)
# canny        
    edge = cv2.Canny(img, 50, 250) 
    resize_edge = cv2.resize(edge,dsize=(67,67),interpolation=cv2.INTER_AREA)
    cv2.imwrite('C:/Users/kjh97/lastprac/rr/1.jpg',resize_edge)
    os.remove('C:/Users/kjh97/lastprac/rr/0.jpg')
# np.list
    mat = glob.glob('C:/Users/kjh97/lastprac/rr/1.jpg')
    img = cv2.imread(mat[0])
    arr = np.array(img)
    list_x = []
    list_x.append(arr)
    result = np.array(list_x)
#load model
    model = load_model('jong_s_pill_last_s87.h5')
    predictions = model.predict(result)
    rs = np.argmax(predictions)
    os.remove('C:/Users/kjh97/lastprac/rr/1.jpg')
    return render(request, "result.html", {'rs':rs})