import numpy as np
import cv2
import os
from matplotlib import pyplot as plt

files = os.listdir('./data/')

faces_path = []
labels = [] # labels for images

#finding image path for every image and label assiciated with image
for file in files:
    images = os.listdir('./data/'+file)
    for im in images:
        faces_path.append('./data/'+file+'/'+im)
        labels.append(file)

#storing the image into an array
images = []

for img_path in faces_path:
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(100,100))
    images.append(img)

print len(images)


all_faces = np.asarray(images)

labels = np.asarray(labels)

#saving the numpy array of the images into numpy file and using this dataset for the image recognition model

np.save('./data/dataset/face_data',all_faces)
np.save('./data/dataset/face_label',labels)

print all_faces.shape
print labels.shape
