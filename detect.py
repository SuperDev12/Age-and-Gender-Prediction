# import cv2
# import math
# import argparse

# def highlightFace(net, frame, conf_threshold=0.7):
#     frameOpencvDnn=frame.copy()
#     frameHeight=frameOpencvDnn.shape[0]
#     frameWidth=frameOpencvDnn.shape[1]
#     blob=cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

#     net.setInput(blob)
#     detections=net.forward()
#     faceBoxes=[]
#     for i in range(detections.shape[2]):
#         confidence=detections[0,0,i,2]
#         if confidence>conf_threshold:
#             x1=int(detections[0,0,i,3]*frameWidth)
#             y1=int(detections[0,0,i,4]*frameHeight)
#             x2=int(detections[0,0,i,5]*frameWidth)
#             y2=int(detections[0,0,i,6]*frameHeight)
#             faceBoxes.append([x1,y1,x2,y2])
#             cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
#     return frameOpencvDnn,faceBoxes


# parser=argparse.ArgumentParser()
# parser.add_argument('--image')

# args=parser.parse_args()

# faceProto="opencv_face_detector.pbtxt"
# faceModel="opencv_face_detector_uint8.pb"
# ageProto="age_deploy.prototxt"
# ageModel="age_net.caffemodel"
# genderProto="gender_deploy.prototxt"
# genderModel="gender_net.caffemodel"

# MODEL_MEAN_VALUES=(78.4263377603, 87.7689143744, 114.895847746)
# ageList=['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
# genderList=['Male','Female']

# faceNet=cv2.dnn.readNet(faceModel,faceProto)
# ageNet=cv2.dnn.readNet(ageModel,ageProto)
# genderNet=cv2.dnn.readNet(genderModel,genderProto)

# video=cv2.VideoCapture(args.image if args.image else 0)
# padding=20
# while cv2.waitKey(1)<0 :
#     hasFrame,frame=video.read()
#     if not hasFrame:
#         cv2.waitKey()
#         break
    
#     resultImg,faceBoxes=highlightFace(faceNet,frame)
#     if not faceBoxes:
#         print("No face detected")

#     for faceBox in faceBoxes:
#         face=frame[max(0,faceBox[1]-padding):
#                    min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
#                    :min(faceBox[2]+padding, frame.shape[1]-1)]

#         blob=cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
#         genderNet.setInput(blob)
#         genderPreds=genderNet.forward()
#         gender=genderList[genderPreds[0].argmax()]
#         print(f'Gender: {gender}')

#         ageNet.setInput(blob)
#         agePreds=ageNet.forward()
#         age=ageList[agePreds[0].argmax()]
#         print(f'Age: {age[1:-1]} years')

#         cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
#         cv2.imshow("Detecting age and gender", resultImg)
import cv2
import argparse
from mtcnn import MTCNN
import numpy as np
import concurrent.futures

def highlightFace(frame, detector, conf_threshold=0.7):
    faceBoxes = []
    faces = detector.detect_faces(frame)
    for face in faces:
        confidence = face["confidence"]
        if confidence > conf_threshold:
            x, y, width, height = face["box"]
            x2, y2 = x + width, y + height
            faceBoxes.append([x, y, x2, y2])
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
    return frame, faceBoxes

def apply_smoothing(current_value, previous_value, alpha=0.7):
    return alpha * current_value + (1 - alpha) * previous_value

def process_frame(frame, face_detector, ageNet, genderNet, ageList, genderList, prev_gender_preds, prev_age_preds, MODEL_MEAN_VALUES):
    padding = 20
    resultImg, faceBoxes = highlightFace(frame, face_detector)
    if not faceBoxes:
        return resultImg
    
    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding):min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding):min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

        genderNet.setInput(blob)
        genderPreds = genderNet.forward()[0]
        smoothed_gender_preds = apply_smoothing(genderPreds, prev_gender_preds)
        gender = genderList[smoothed_gender_preds.argmax()]
        prev_gender_preds = smoothed_gender_preds

        ageNet.setInput(blob)
        agePreds = ageNet.forward()[0]
        smoothed_age_preds = apply_smoothing(agePreds, prev_age_preds)
        age = ageList[smoothed_age_preds.argmax()]
        prev_age_preds = smoothed_age_preds

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    
    return resultImg

def main(args):
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)
    face_detector = MTCNN()

    video = cv2.VideoCapture(args.image if args.image else 0)

    # Initialize smoothing variables
    prev_gender_preds = np.zeros((2,))
    prev_age_preds = np.zeros((8,))

    # Asynchronous video processing
    with concurrent.futures.ThreadPoolExecutor() as executor:
        while cv2.waitKey(1) < 0:
            hasFrame, frame = video.read()
            if not hasFrame:
                cv2.waitKey()
                break
            
            # Submit frame processing to the executor
            future = executor.submit(process_frame, frame, face_detector, ageNet, genderNet, ageList, genderList, prev_gender_preds, prev_age_preds, MODEL_MEAN_VALUES)
            resultImg = future.result()  # Get the processed frame
            
            cv2.imshow("Detecting age and gender", resultImg)

    # Release video capture and close windows
    video.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--image')
    args = parser.parse_args()
    main(args)
