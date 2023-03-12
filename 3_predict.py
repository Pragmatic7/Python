import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras
import numpy as np
import os,glob,cv2
import sys,argparse

print(tf.__version__)


new_model = tf.keras.models.load_model('saved_model/MaskContaminationModel')

# Check its architecture
new_model.summary()

# face_cascade = cv2.CascadeClassifier('/usr/share/opencv/haarcascades/haarcascade_frontalface_default.xml')
os.chdir("MaskHSHair/Marginal")

files = sorted(glob.glob('*.jpg'))
# files.extend(sorted(glob.glob('*.jpg')))
# files.extend(sorted(glob.glob('*.png')))
totalNumber = len(files)
print("total number of images is:", totalNumber)

num_classes = 2
image_size = 180
num_channels = 3
# images = np.zeros(image_size,image_size,3*totalNumber)
text_file = open("C:/Users/Downloads/Deep learning/MaskContamination/TF231/version5/Results_MaskHSHair_Marginal.txt", "w")

indx = 0
i = 0
probability_model = tf.keras.Sequential([new_model, tf.keras.layers.Softmax()])

for file in files:
	indx = indx + 1
	images = []
	print("processing image %d out of %d " %(indx,len(files)))
	print(file)
	text_file.write(file)
	# Reading the image using OpenCV
	image = cv2.imread(file)
	# face detection
	# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
	# if len(faces) < 1:
 #             print("No faces found. Skipping this images")
 #             text_file.write("\n")
 #             text_file.write("NA")
 #             text_file.write("\n")
 #             continue
	# for (xx,yy,ww,hh) in faces:
	#     image = cv2.rectangle(image,(xx,yy),(xx+ww,yy+hh),(255,0,0),2)
	#     roi_color = image[yy:yy+hh, xx:xx+ww]
	# Resizing the image to our desired size
	# and preprocessing will be done exactly as done during training
	image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
	images.append(image)
	images = np.array(images, dtype=np.uint8)
	images = images.astype('float32')



# Should NOT normalize the data since 1st layer in model ALREADY DOING THAT!



	# images = np.multiply(images, 1.0/255.0) 
	# print(images.shape)
	#The input to the network is of shape
	# [None image_size image_size num_channels]. Hence we reshape.
	x_batch = images.reshape(1, image_size,image_size,num_channels)

	scores = new_model.predict(x_batch)
	print(scores)

	# predictions = probability_model.predict(x_batch)
	# print(predictions)



	# predicts = new_model.predict(x_batch)
	# print(predicts)

	
	text_file.write("\n")
	for j in range(num_classes):
		text_file.write('%s' % scores[i,j])
		text_file.write("\t")
	text_file.write("\n")	

text_file.close()
