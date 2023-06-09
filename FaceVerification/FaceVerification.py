import logging
import os

# Supress warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)

from tensorflow import keras 
from tensorflow.keras.models import load_model
import mtcnn
from PIL import Image
import numpy as np
from numpy import expand_dims
from numpy import asarray
from mtcnn.mtcnn import MTCNN
from os import listdir

user_args = sys.argv[1:]

image_path1, image_path2, modelPath = user_args
# print("Received User Args.")

# load the model
# model = load_model('facenet_keras.h5')
model = load_model(modelPath)
# print("Loaded Model.")

# function for face detection with mtcnn
# extract a single face from a given photograph
def extract_face(filename, required_size=(160, 160)):
    # load image from file
    image = Image.open(filename)
    # convert to RGB, if needed
    image = image.convert('RGB')
    # convert to array
    pixels = asarray(image)
    # create the detector, using default weights
    detector = MTCNN()
    # detect faces in the image
    results = detector.detect_faces(pixels)
    # extract the bounding box from the first face
    x1, y1, width, height = results[0]['box']
    # bug fix
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    # extract the face
    face = pixels[y1:y2, x1:x2]
    # resize pixels to the model size
    image = Image.fromarray(face)
    image = image.resize(required_size)
    face_array = asarray(image)
    return face_array

# load the photo and extract the face
facePixels1 = extract_face(image_path1)
facePixels2 = extract_face(image_path2)
# print("Extracted Faces.")

###################### Face Embedding #################
# get the face embedding for one face
def get_embedding(model, face_pixels):
    # scale pixel values
    face_pixels = face_pixels.astype('float32')
    # standardize pixel values across channels (global)
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    # transform face into one sample
    samples = expand_dims(face_pixels, axis=0)
    # make prediction to get embedding
    yhat = model.predict(samples)
    return yhat[0]

embedding1 = get_embedding(model, facePixels1)
embedding2 = get_embedding(model, facePixels2)
# print("Calculated Face Embeddings.")

# Calculate Eucledian distance between two embedding vectors
def distance(v1, v2):
    return np.sqrt(np.sum((v1 - v2) ** 2))  

euclideanDistance = distance(embedding1, embedding2)
# This print will return euclideanDistance value to java as string
# print("Here is the Euclidean Distance:")
print(euclideanDistance)
