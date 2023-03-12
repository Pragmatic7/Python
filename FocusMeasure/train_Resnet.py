
import numpy as np
import os
import PIL
import PIL.Image
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import layers
import pathlib
import shutil
from tensorflow.keras.applications import ResNet50
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, GlobalAveragePooling2D


print(tf.__version__)

numberOfEpochs = 60
batch_size = 30
img_height = 224
img_width = 224
num_classes = 2


data_dir = 'C:/Users/Downloads/Deep learning/FocusDL/Data/Dataset/Training'

data_dir = pathlib.Path(data_dir)
print(data_dir)

image_count = len(list(data_dir.glob('*/*.jpg')))
print(image_count)

train_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  shuffle=True,
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="validation",
  shuffle=True,
  seed=1,
  image_size=(img_height, img_width),
  batch_size=batch_size)

class_names = train_ds.class_names
print(class_names)

for image_batch, labels_batch in train_ds:
  print(image_batch.shape)
  print(labels_batch.shape)
  break


AUTOTUNE = tf.data.experimental.AUTOTUNE

train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)


# ### Train a model ResNet50
base_model = ResNet50(input_shape=(224, 224,3), include_top=False, weights="imagenet")

for layer in base_model.layers:
  layer.trainable = False

base_model = Sequential()
base_model.add(ResNet50(include_top=False, weights='imagenet', pooling='max'))
base_model.add(Dense(1, activation='sigmoid'))

base_model.compile(optimizer = tf.keras.optimizers.SGD(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])

base_model.build(image_batch.shape) # `input_shape` is the shape of the input data
                         # e.g. input_shape = (None, 32, 32, 3)
base_model.summary()

# Include the epoch in the file name (uses `str.format`)
checkpoint_path = "checkpoints_Resnet/cp-{epoch:04d}.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)
shutil.rmtree(checkpoint_dir)

# Save checkpoints during training
# Create a callback that saves the model's weights every 5 epochs
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_path, 
    monitor='val_accuracy', 
    verbose=2, 
    save_weights_only=True,
    mode='max',
    save_best_only=True)


# Save the weights using the `checkpoint_path` format
base_model.save_weights(checkpoint_path.format(epoch=0))

# Train the model with the new callback
base_model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=numberOfEpochs,
  callbacks=[cp_callback],
  verbose=2
)

# Save the entire model
modelSavePath = 'saved_model_Resnet'
shutil.rmtree(modelSavePath)


base_model.save('saved_model_Resnet/model_focus')
