import cv2
import numpy as np
import os 
import writer




def facerecognition():
    
    camera_port = 1
        
    etudiants = {
        1 : "Errouk",
        2 : "Bentouhami",
        3 : "Mouzafir",
        4 : "Tamega",
        5 : "Firoud",
        6 : "Oubenaddi",
        17 : "Aminatou",
        8 : "Marouni",
        9 : "Aguenchich",
        10 : "Oussahi",
        11 : "Kannoufa",
        12 : "Outhouna",
        13 : "Ougoud",
        16 :"sami",
        14 :"sami"
    }

    recognizer = cv2.face.LBPHFaceRecognizer_create()

    recognizer.read('saved_model1/s_model.yml')

    faceCascade = cv2.CascadeClassifier("face_detection.xml")

    #cam = cv2.VideoCapture(camera_port,cv2.CAP_DSHOW)

    #while True:
    imagePaths=[os.path.join("./folder",f) for f in os.listdir("./folder")]
    for imagePath in imagePaths:
        print(imagePath)
        #ret, img =cam.read()
        img = cv2.imread(imagePath).copy()
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        cv2.imshow('frame',img);
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.5,
                                    minNeighbors = 5,
                                    minSize= (30,30))

        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x-20,y-20), (x+w+20,y+h+20), (0,255,0), 2)

            Id, confidence = recognizer.predict(gray[y:y+h,x:x+w])

            if((100 - confidence)>10):
                label = etudiants[Id] + " {0:.2f}%".format(round(100 - confidence, 2))
                
                cv2.rectangle(img, (x-20,y-85), (x+w+20, y-20), (0,255,0), -1)
                cv2.putText(img, label, (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                writer.output('Presence_IRISI2_', 'IRISI 2', Id, etudiants[Id], 'yes')
            else :
                cv2.putText(img,"non reconnu" , (x,y-40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
                
        #cv2.imshow('face recognition', img) 

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
        
        while(True):
            cv2.imshow('frame',img);
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break;

    #cam.release()

    cv2.destroyAllWindows()


facerecognition()