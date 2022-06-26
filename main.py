from flask import Flask, render_template, Response
from numpy import rint
from requestCamera import VideoCamera
import os
import glob

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

def gen(camera):
    while True:
        frame = camera.get_frame()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

@app.route('/fer')
def fer():
    list = glob.glob("src/test/*.mp4")
    print(list)
    return Response(gen(VideoCamera(list)), mimetype='multipart/x-mixed-replace; boundary=frame')
   

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)

# this is main file