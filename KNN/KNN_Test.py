import numpy as np
import cv2
import os
from capture import max_images
# max_images=100

# instantiate the camera object and haar cascade
cam = cv2.VideoCapture(0)
face_cas = cv2.CascadeClassifier(r'C:\Users\BCNQ8456\AppData\Local\Continuum\anaconda3\envs\FaceRecognitionEnv\Library\etc\haarcascades\haarcascade_frontalface_default.xml')
# declare the type of font to be used on output window
font = cv2.FONT_HERSHEY_COMPLEX

# create a look-up dictionary
names = {}
data = []
# load the data from the numpy matrices and convert to linear vectors
persons = os.listdir('dataSet/')
for person in persons:
    print('loading face '+person[0:-4].title())
    f = np.load('dataSet/'+person).reshape((max_images, 50*50*3))
    names[len(names)] = person[:-4].title()
    data.append(f)
# combine all info into one data array
data = np.concatenate(data) # (60, 7500)

# create a matrix to store the labels
labels = np.zeros((len(data), 1))
for i in names.keys():
    labels[i*max_images:(i+1)*max_images, :] = i

print(data.shape, labels.shape) # (60, 1)

# the distance and knn functions we defined earlier
def distance(x1, x2):
    return np.sqrt(((x1-x2)**2).sum())

def knn(x, train, targets, k=5):
    m = train.shape[0]
    dist = []
    for ix in range(m):
        # compute distance from each point and store in dist
        dist.append(distance(x, train[ix]))
    dist = np.asarray(dist)
    indx = np.argsort(dist)
    sorted_labels = labels[indx][:k]
    counts = np.unique(sorted_labels, return_counts=True)
    return counts[0][np.argmax(counts[1])]

while True:
    # get each frame
    ret, frame = cam.read()
    if ret == True:
        # convert rgb to grayscale and get faces
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cas.detectMultiScale(gray, 1.3, 5)
        # for each face
        for (x, y, w, h) in faces:
            face_component = frame[y:y+h, x:x+w, :]
            fc = cv2.resize(face_component, (50, 50))
            # after processing the image and rescaling
            # convert to linear vector using .flatten()
            # and pass to knn function along with all the data
            lab = knn(fc.flatten(), data, labels)
            # convert this label to int and get the corresponding name
            text = names[int(lab)]
            # display the name
            cv2.putText(frame, text, (x, y), font, 1, (255, 255, 0), 2)
            # draw a rectangle over the face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
            cv2.imshow('face recognition', frame)
        if cv2.waitKey(1) == 27: #break on escape key
            break
    else:
        print('Error')
cv2.destroyAllWindows()