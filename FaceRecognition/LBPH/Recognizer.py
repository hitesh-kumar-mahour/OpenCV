import numpy as np
import cv2
import sqlite3
def recognizeperson(Id):
    conn=sqlite3.connect("pdb.sqlite")
    c=conn.cursor()
    conn.commit()
    tempe=[]
    for val in c.execute('SELECT * from people where Id=?',(id,)):
        tempe.append(val)
    conn.commit()
    conn.close()
    name=tempe[0][1]
    username=tempe[0][2]
    mobile =tempe[0][4]
    email=tempe[0][3]
    st="Recognized person : NAME: "+name+" ,Username: "+username+" ,MOBILE: "+mobile+" and EMAIL: "+email
    return (st,name)
if __name__ == "__main__":
    faceDetect = cv2.CascadeClassifier(r'C:\Users\BCNQ8456\AppData\Local\Continuum\anaconda3\envs\FaceRecognitionEnv\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
    cam=cv2.VideoCapture(0)
    rec=cv2.face.LBPHFaceRecognizer_create()
    rec.read('trainingData.yml')
    font=cv2.FONT_HERSHEY_COMPLEX_SMALL
    try:
        while(True):
            ret, img = cam.read()
            if ret:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                faces=faceDetect.detectMultiScale(gray, 1.3, 5)
                for (x,y,w,h) in faces:
                    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                    id,conf=rec.predict(gray[y:y+h,x:x+w])
                    stat,name=recognizeperson(id)
                    print(stat)
                    cv2.putText(img,name+' '+str(conf)[:4],(x,y+h),font,1,(0,0,255),2)
                cv2.imshow("Face",img)
                if(cv2.waitKey(1)==27):
                    break
    except:
        cam.release()
    cam.release()
    cv2.destroyAllWindows()