from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread

global capture, rec_frame, grey, switch, neg, face, eyes, rec, out 
capture = 0
grey = 0
neg = 0
face = 0
eyes = 0
switch = 1
rec = 0

# Making directory "saved" to save pics and videos
try: os.mkdir('./saved')
except OSError as error: pass

# Loading pretrained face detection model    
faceClass = cv2.CascadeClassifier('assets/haarcascade_frontalface_default.xml')
eyeClass = cv2.CascadeClassifier('assets/haarcascade_eye.xml')

# Instatiate flask app  
app = Flask(__name__, template_folder='./templates')

# Capture video through webcam
camera = cv2.VideoCapture(0)

# Recording video
def record(out):
    global rec_frame
    while(rec):
        time.sleep(0.05)
        out.write(rec_frame)

# Face detection
def detect_face(frame):
    global faceClass

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    face = faceClass.detectMultiScale(gray , 1.5 , 5)

    for (x , y , w , h) in face:
        cv2.rectangle(frame , (x , y) , (x + w , y + h) , (255 , 0 , 0) , 3)

    return frame

# Eyes detection
def detect_eyes(frame):
    global eyeClass

    gray = cv2.cvtColor(frame , cv2.COLOR_BGR2GRAY)
    eye = eyeClass.detectMultiScale(gray , 1.5 , 20)

    for (x , y , w , h) in eye:
        cv2.rectangle(frame , (x , y) , (x + w , y + h) , (0 , 255 , 0) , 3)

    return frame
 

def gen_frames():  # Generate frames from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            if(face):                
                frame= detect_face(frame)
            if(eyes):                
                frame= detect_eyes(frame)
            if(grey):
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if(neg):
                frame=cv2.bitwise_not(frame)    
            if(capture):
                capture=0
                now = datetime.datetime.now()
                p = os.path.sep.join(['saved', "shot_{}.png".format(str(now).replace(":",''))])
                cv2.imwrite(p, frame)
            
            if(rec):
                rec_frame=frame
                frame= cv2.putText(cv2.flip(frame,1),"Recording...", (0,25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255),4)
                frame=cv2.flip(frame,1)

            try:
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e: pass
      
        else: pass


@app.route('/')
def index():
    return render_template('index.html')
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/requests',methods=['POST','GET'])
def tasks():
    global switch,camera
    if request.method == 'POST':
        if request.form.get('click') == 'Capture':
            global capture
            capture=1
        elif  request.form.get('grey') == 'Grey':
            global grey
            grey=not grey
        elif  request.form.get('neg') == 'Negative':
            global neg
            neg=not neg
        elif  request.form.get('face') == 'Face Detection':
            global face
            face=not face 
            if(face):
                time.sleep(4)   
        elif  request.form.get('eyes') == 'Eyes Detection':
            global eyes
            eyes=not eyes 
            if(eyes):
                time.sleep(4) 
        elif  request.form.get('stop') == 'Start/Stop':
            
            if(switch==1):
                switch=0
                camera.release()
                cv2.destroyAllWindows()
                
            else:
                camera = cv2.VideoCapture(0)
                switch=1
        elif  request.form.get('rec') == 'Record':
            global rec, out
            rec= not rec
            if(rec):
                now=datetime.datetime.now() 
                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                out = cv2.VideoWriter('./saved/vid_{}.mp4'.format(str(now).replace(":",'')), fourcc, 20.0, (640, 480))
                # Start new thread for recording the video
                thread = Thread(target = record, args=[out,])
                thread.start()
            elif(rec==False):
                out.release()
                          
                 
    elif request.method=='GET':
        return render_template('index.html')
    return render_template('index.html')


if __name__ == '__main__':
    app.run()

camera.release()
cv2.destroyAllWindows()     