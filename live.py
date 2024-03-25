
# Since I could not use the emotion model, I took it to the comment line, you can use it by deleting the comment line.

# from tensorflow.keras.preprocessing.image import img_to_array #Can use cv2 or other libraries.
import cv2
from tflite_runtime.interpreter import Interpreter

import numpy as np
import time

face_classifier=cv2.CascadeClassifier('haarcascades_models/haarcascade_frontalface_default.xml')

#### PREDICT USING tflite ###
#tflite with optimization is taking too long on Windows, so not even try.
#On Jetson you can try both opt and no opt. 

# Load the TFLite model and allocate tensors.
"""emotion_interpreter = Interpreter(model_path="emotion_detection_model_100epochs_opt.tflite")
emotion_interpreter.allocate_tensors()"""

age_interpreter = Interpreter(model_path="./pre-trained-models/age_detection_model_100epochs_opt.tflite")
age_interpreter.allocate_tensors()

gender_interpreter = Interpreter(model_path="./pre-trained-models/gender_detection_model_100epochs_opt.tflite")
gender_interpreter.allocate_tensors()


# Get input and output tensors.
"""emotion_input_details = emotion_interpreter.get_input_details()
emotion_output_details = emotion_interpreter.get_output_details()"""

age_input_details = age_interpreter.get_input_details()
age_output_details = age_interpreter.get_output_details()

gender_input_details = gender_interpreter.get_input_details()
gender_output_details = gender_interpreter.get_output_details()


# Test the model on input data.
"""emotion_input_shape = emotion_input_details[0]['shape']"""
age_input_shape = age_input_details[0]['shape']
gender_input_shape = gender_input_details[0]['shape']

###########################


"""class_labels=['Angry','Disgust', 'Fear', 'Happy','Neutral','Sad','Surprise']"""
gender_labels = ['Male', 'Female']

def gstreamer_pipeline(
    capture_width=640,
    capture_height=480,
    display_width=640,
    display_height=480,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        "width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink drop=True"
        % (
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )


#cap=cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

cap = cv2.VideoCapture("nvarguscamerasrc ! video/x-raw(memory:NVMM), width=(int)1280, height=(int)720, format=(string)NV12, framerate=(fraction)30/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink")


while True:
    ret,frame=cap.read()
    labels=[]
    
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    start = time.time()

    """ for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray=gray[y:y+h,x:x+w]
        roi_gray=cv2.resize(roi_gray,(48,48),interpolation=cv2.INTER_AREA)

        #Get image ready for prediction
        roi=roi_gray.astype('float')/255.0  #Scale
        #roi=img_to_array(roi)
        roi=np.expand_dims(roi,axis=0)  #Expand dims to get it ready for prediction (1, 48, 48, 1)
        
        #emotion_interpreter.set_tensor(emotion_input_details[0]['index'], roi)
        #emotion_interpreter.invoke()
        #emotion_preds = emotion_interpreter.get_tensor(emotion_output_details[0]['index'])

        #preds=emotion_model.predict(roi)[0]  #Yields one hot encoded result for 7 classes
        #emotion_label=class_labels[emotion_preds.argmax()]  #Find the label
        #emotion_label_position=(x,y)
        #cv2.putText(frame,emotion_label,emotion_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2) 

        
        
        roi_color=frame[y:y+h,x:x+w]
        roi_color=cv2.resize(roi_color,(200,200),interpolation=cv2.INTER_AREA)
        roi_color = np.array(roi_color).reshape(-1,200,200,3) #input shape is (1, 200,200,3)
        roi_color = roi_color.astype(np.float32)
        
        gender_interpreter.set_tensor(gender_input_details[0]['index'], roi_color)
        gender_interpreter.invoke()
        gender_preds = gender_interpreter.get_tensor(gender_output_details[0]['index'])
        
        
        #gender_predict = gender_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        gender_predict = (gender_preds>= 0.5).astype(int)[:,0]
        gender_label=gender_labels[gender_predict[0]] 
        gender_label_position=(x,y+h+50) #50 pixels below to move the label outside the face
        cv2.putText(frame,gender_label,gender_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
        #Age
        
        age_interpreter.set_tensor(age_input_details[0]['index'], roi_color)
        age_interpreter.invoke()
        age_preds = age_interpreter.get_tensor(age_output_details[0]['index'])
        
        #age_predict = age_model.predict(np.array(roi_color).reshape(-1,200,200,3))
        age = round(age_preds[0,0])
        age_label_position=(x+h,y+h)
        cv2.putText(frame,"Age="+str(age),age_label_position,cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        
    end=time.time()
    print("Total time=", end-start) """
    
    cv2.imshow('Emotion Detector', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):  #Press q to exit
        break
cap.release()
cv2.destroyAllWindows()