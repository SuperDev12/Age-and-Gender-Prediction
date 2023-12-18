from flask import Flask, render_template, request
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from io import BytesIO
import base64

app = Flask(__name__)

gender_model = cv2.dnn.readNetFromCaffe("deploy_gender.prototxt", "gender_net.caffemodel")
age_model = cv2.dnn.readNetFromCaffe("deploy_age.prototxt", "age_net.caffemodel")
df = pd.read_csv('age_gender.csv')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

def predict_age_gender(face):
    blob_gender = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    gender_model.setInput(blob_gender)
    gender_preds = gender_model.forward()

    blob_age = cv2.dnn.blobFromImage(face, scalefactor=1.0, size=(227, 227), mean=(78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
    age_model.setInput(blob_age)
    age_preds = age_model.forward()

    gender = "Male" if gender_preds[0][0] > 0.5 else "Female"

    age_preds = age_preds[0]
    age_class = np.argmax(age_preds)
    age = (age_class + 1) * 5

    return gender, age

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files['file']
        
        if file:
            image = cv2.imdecode(np.fromstring(file.read(), np.uint8), cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gender, age = predict_age_gender(image_rgb)
            result_text = f'Gender: {gender}, Age: {age} years'
            _, img_encoded = cv2.imencode('.png', cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
            img_base64 = "data:image/png;base64," + base64.b64encode(img_encoded).decode('utf-8')
            return render_template("index.html", result=result_text, image=img_base64)

    return render_template("index.html", result=None, image=None)

if __name__ == "__main__":
    app.run(debug=True)
