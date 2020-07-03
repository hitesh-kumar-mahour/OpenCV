from PIL import Image
import cv2,os
import numpy as np
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'
# name_dict={}
def getImageswithId(path):
    imagePaths=[os.path.join(path,f) for f in os.listdir(path)]
    faces=[]
    IDs=[]
    for imagePath in imagePaths:
        faceImg=Image.open(imagePath).convert('L')
        faceNp=np.array(faceImg,'uint8')
        # name = os.path.split(imagePath)[-1].split('_')[0]
        # if not name_dict.__contains__(name):
        #     name_dict=len(name_dict)
        ID=int(os.path.split(imagePath)[-1].split('_')[0])
        faces.append(faceNp)
        IDs.append(ID)
        cv2.imshow("Training.",faceNp)
        cv2.waitKey(10)
    cv2.destroyAllWindows()
    return np.array(IDs),faces
Ids,faces=getImageswithId(path)
recognizer.train(faces,Ids)
recognizer.write('trainingData.yml')
print('Training Complete')
