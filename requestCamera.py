import cv2
import numpy as np
from model import FacialExpressionModel
from csv import writer
import pandas as pd

model = FacialExpressionModel()

facec = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
# filename='src/2.mp4'
all_emo = ["Angry", "Disgust", "Fear", "Happy", "Neutral", "Sad", "Surprise"] 
all_em = [0, 1, 2, 3, 6, 5, 4] 

class VideoCamera(object):
    def __init__(self, list):
        self.list = list
    def __del__(self):
        self.video.release()

    def get_frame(self):
        columns=['filename', 'emotion']
        data=[]
        for val in self.list:
            self.video = cv2.VideoCapture(val)
            hasFrame, fr = self.video.read()
            gray_fr = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
            faces = facec.detectMultiScale(gray_fr, 1.3, 10)
            
            for (x, y, w, h) in faces:
                fc = gray_fr[y:y+h, x:x+w]
                roi = cv2.resize(fc, (48, 48))
                text = model.predict_emotion(roi[np.newaxis, :, :, np.newaxis])
                cv2.putText(fr, text, (x, y + h), font, 1, (0, 0, 255), 2)
                cv2.rectangle(fr, (x, y),(x + w, y + h),(0,255,0),2)
            for value in all_emo:
                if(text==value):
                    num=all_emo.index(value)
            list=[val.replace("src/test\\", ''), num]
            data.append(list)
        df = pd.DataFrame(data, columns=columns)
        df.to_csv('res.csv',mode = 'w', index=False) 
            

