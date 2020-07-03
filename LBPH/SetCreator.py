# for LBPH implementation, as we have connected our dataset
# with a database to provide
# information of the recognised people we have used sqlite3 DB
import cv2
import sqlite3
max_images=101

def createtable():
    connection = sqlite3.connect('pdb.sqlite')
    cursor = connection.cursor()
    cursor.execute('CREATE TABLE people (Id INTEGER PRIMARY KEY AUTOINCREMENT, Name TEXT, Username TEXT, Email TEXT,Mobile TEXT);')
    # connection.commit()
    # connection.close()
    print('Table created successfully')

def check(a):
    conn=sqlite3.connect("pdb.sqlite")
    c=conn.cursor()
    tempe=[]
    for val in c.execute('SELECT * from people WHERE Username=?',(a,)):
        tempe.append(val)
    conn.close()
    if len(tempe)==0:
        return True
    else:
        return False

def counttablerows():
    conn=sqlite3.connect('pdb.sqlite')
    c=conn.cursor()
    conn.commit()
    count=0
    for val in c.execute("select count(Name) from people"):
        count=int(val[0])
    conn.commit()
    conn.close()
    return count

def adddata(b,f,d,e):
    inicount=counttablerows()
    conn=sqlite3.connect("pdb.sqlite")
    c=conn.cursor()
    conn.commit()
    c.execute('INSERT INTO people (Name,Username,Email,Mobile) VALUES (?,?,?,?)',(b,f,d,e))
    conn.commit()
    fincount=counttablerows()
    if fincount==(inicount+1):
        return "OK"
    else:
        return "ERROR"

def getId(a):
    conn=sqlite3.connect("pdb.sqlite")
    c=conn.cursor()
    conn.commit()
    tempe=[]
    for val in c.execute('SELECT * from people WHERE Username=?',(a,)):
        tempe.append(val)
    identity=tempe[0][0]
    return (identity)

def takedata():
    name=""
    username=""
    mobile=""
    email=""
    while True:
        name=input("Enter Your Name: ").strip()
        while True:
            username=input("Enter Username: ").strip()
            if check(username):
                break
            else:
                print("Username",username+" already exists.")
                continue
        mobile=input("Enter your Mobile Number: ").strip()
        email=input("Enter your Email ID: ").strip()
        res=adddata(name,username,email,mobile)
        if res=="OK":
            print("DATA Added Successfully.")
            break
        else:
            print("Error Occured")
            continue
    return getId(username)


# Choose the Video Camera to initialize
cam = cv2.VideoCapture(0)

# Adding classifier - Haarcascade xml file
detector = cv2.CascadeClassifier(r'C:\Users\BCNQ8456\AppData\Local\Continuum\anaconda3\envs\FaceRecognitionEnv\Library\etc\haarcascades\haarcascade_frontalface_default.xml')

#creating table
try:
    createtable()
except:
    print('Table already present')

# Taking Data from user
# print("Enter your name:")
Id=takedata()
sampleNum=0
# Taking a set of 21 pictures of the person.
try:
    while(True):
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.3, 5)
        for (x,y,w,h) in faces:
            cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
            #incrementing sample number
            sampleNum=sampleNum+1
            #saving the captured face in the dataset folder
            # cv2.imwrite("dataSet/User." + str(Id) + '.' + str(sampleNum) + ".jpg"
            # 'dataSet/' + str(Id) + '_' + str(sampleNum) + ".jpg"
            if cv2.imwrite('dataSet/' + str(Id) + '_' + str(sampleNum) + ".jpg",gray[y:y+h,x:x+w]):
                print("Saved "+str(Id)+'_'+ str(sampleNum) +".jpg")
        cv2.imshow('SetCreator',img)
        #wait for 100 miliseconds
        # if the user presses the escape key (ID: 27) or sample number is morethan max_images, we stop recording.
        if cv2.waitKey(100) == 27 or sampleNum >= max_images:
            break

except:
    cam.release()
    print("Exception caught")
cam.release()
cv2.destroyAllWindows()
print("Dataset Created successfully..")