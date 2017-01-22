import numpy as np
from matplotlib import pyplot as plt
import cv2
from knn import knn

rgb = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

a = np.random.random

font = cv2.FONT_HERSHEY_COMPLEX

def recongnise_face(img):
    img = cv2.resize(img , (100,100))
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # print img.shape , img
    img_flatten = img.flatten()
    image_data = np.load('./data/dataset/face_data.npy')
    image_data.reshape((16,10000))
    images_train = []
    for i in range(image_data.shape[0]):
        img = image_data[i]
        img = np.asarray(img)
        img = img.flatten()
        print img.shape
        images_train.append(img)
    #image_data = image_data.flatten()
    image_labels = np.load('./data/dataset/face_label.npy')
    image_labels = image_labels.flatten()

    person = knn(images_train,image_labels,img_flatten,k=4)

    return person
#rgb.open()
while True:
    ret , fr = rgb.read()
    print ret
    gray = cv2.cvtColor(fr,cv2.COLOR_BGR2GRAY)

    face = face_cascade.detectMultiScale(gray,1.3,5)

    for(x,y,w,h) in face:
        fc = fr[x:x+w,y:y+h,:]
        fc2 = fr[y:y+h,x:x+w,:]
        out = recongnise_face(fc)[0]

        print out

        cv2.putText(fr,out,(int(x),int(y)),fontFace=font,fontScale=3,color=(255,0,0))
        cv2.rectangle(fr,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("rgb",fr)
    cv2.imshow("gray",gray)

    if cv2.waitKey(1) == 27 : # 1 signifies amunt of time to hold in milisecond
        break


cv2.destroyAllWindows()
