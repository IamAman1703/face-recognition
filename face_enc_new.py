

from imutils import paths
import face_recognition
import pickle
import cv2
import os
import numpy as np

dataset= r'C:\Users\rajpu\Desktop\ai'

args= {}
args['dataset']= dataset


iPaths = list(paths.list_images(args["dataset"]))  #image paths
data = []
labels = []
for iPath in iPaths:
    label = iPath.split(os.path.sep)[-2]   #split the image paths
    image = cv2.imread(iPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) #Convert images into RGB Channel
    image = cv2.resize(image, (224, 224))  #Resizing the images
    data.append(image)
    labels.append(label)
data = np.array(data) / 255.0
labels = np.array(labels)

knownNames = labels

data_new= {"encodings": data, "names": knownNames}
#use pickle to save data into a file for later use
f = open("face_enc_new", "wb")
f.write(pickle.dumps(data_new))
f.close()