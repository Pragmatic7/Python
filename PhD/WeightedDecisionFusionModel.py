from __future__ import print_function

import os
import time
import tensorflow as tf
import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import cv2
import numpy as np
import math
import h5py
from batch_handling import batch_gen 
import operator

# Parameters

# path of the training TFRecord
TRAIN_FILE = '/home/user/Downloads/myShearletExperiments/Data/behzad/shearletAllTrainFold5.tfrecords'
# path of the validation TFRecord (is nit used as of now, since we use HDF5 for validation phase)
VALIDATION_FILE = '/home/user/Downloads/myShearletExperiments/Data/behzad/shearletAllTestFold5.tfrecords'
# path of the validation TFRecord (created by the Shearlet_convert_to_records_ALL.py in Data folder)
validation_HDF5='/home/user/Downloads/myShearletExperiments/Data/behzad/test.h5'


batch_size =30
batch_size_val=15
num_epochs = 40
num_train_samples=7280
act_fn = 'relu' 
lr = 0.0001 # learning rate

# reading HDF5 files
h5f = h5py.File(validation_HDF5, 'r')

# converting HDF5 to nd-array
X_test = h5f['X'][()]
Y_test = h5f['Y'][()]
h5f.close()

# converting to one-hot encoding
Y_test = tflearn.data_utils.to_categorical(Y_test, 4)
Y_test = Y_test.astype(np.uint8)

# creating objects to handle batching in validation phase
ob = batch_gen(X = X_test[0:(X_test.shape[0]/2)-2,:,:,:], y = Y_test[0:(X_test.shape[0]/2)-2,:], batch_size = batch_size_val,
                random_init_state= 1)
ob2 = batch_gen(X = X_test[X_test.shape[0]/2:X_test.shape[0],:,:,:], y = Y_test[X_test.shape[0]/2:X_test.shape[0],:], batch_size = batch_size_val,
                random_init_state= 1)

num_iter = X_test.shape[0]/batch_size_val


def weight_variable(shape, varname):
  initial = tf.truncated_normal(shape, stddev=0.1)
  return tf.Variable(initial, name=varname)

def bias_variable(shape, varname):
  initial = tf.constant(0.1, shape=shape)
  return tf.Variable(initial, name=varname)

# This function is used to see the tensors in TensorBoard
def variable_summaries(var):
  """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
  with tf.name_scope('summaries'):
    mean = tf.reduce_mean(var)
    tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
      stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
    tf.summary.scalar('stddev', stddev)
    tf.summary.scalar('max', tf.reduce_max(var))
    tf.summary.scalar('min', tf.reduce_min(var))
    tf.summary.histogram('histogram', var)

# to read data from TFRecord (it is called in the inputs function)
def read_and_decode(filename_queue):
  reader = tf.TFRecordReader()
  _, serialized_example = reader.read(filename_queue)
  features = tf.parse_single_example(
      serialized_example,
      # Defaults are not specified since both keys are required.
      features={
          'train/image': tf.FixedLenFeature([], tf.string),
          'train/label': tf.FixedLenFeature([], tf.int64),
      })

  # Convert from a scalar string tensor
  image = tf.decode_raw(features['train/image'], tf.uint8)
  reshaped_image=tf.reshape(image, [120,120,107])

  height = 120
  width = 120
  channels=107

  num_classes=4


  # Convert from [0, 255] -> [0, 1] floats.
  distorted_image = tf.cast(reshaped_image, tf.float32) * (1. / 255.)

  # Convert label from a scalar uint8 tensor to an int32 scalar.
  label = tf.cast(features['train/label'], tf.int32)
  # label=tf.reshape(label, [7])

  label_categorical = tf.one_hot(
    label,
    depth= num_classes,
    on_value=1,
    off_value=0,
    # axis=None,
    dtype=tf.int32,
  )
  label_categorical = tf.reshape(label_categorical, [num_classes])
  # print (label_categorical.shape)

  # Set the shapes of tensors.
  distorted_image.set_shape([height, width, channels])
  label_categorical.set_shape([num_classes])
  # print (label_categorical.shape)


  return distorted_image, label_categorical

# to read data from TFRecord
def inputs(train, batch_size, num_epochs):
  """Reads input data num_epochs times.

  Args:
    train: Selects between the training (True) and validation (False) data.
    batch_size: Number of examples per returned batch.
    num_epochs: Number of times to read the input data, or 0/None to
       train forever.

  Returns:
    A tuple (images, labels), where:
    * images is a float tensor with shape [batch_size, mnist.IMAGE_PIXELS]
      in the range [-0.5, 0.5].
    * labels is an int32 tensor with shape [batch_size] with the true label,
      a number in the range [0, mnist.NUM_CLASSES).
    Note that an tf.train.QueueRunner is added to the graph, which
    must be run using e.g. tf.train.start_queue_runners().
  """
  if not num_epochs: num_epochs = None
  filename = os.path.join('.',
                          TRAIN_FILE if train else VALIDATION_FILE)
  if os.path.isfile(filename):
    print (filename)

  with tf.name_scope('input'):
    filename_queue = tf.train.string_input_producer(
        [filename], num_epochs = num_epochs)
    # Even when reading in multiple threads, share the filename
    # queue.
    image, label = read_and_decode(filename_queue)
    print (image,label)
    print ('Filling queue with 100 images before starting to train. '
         'This will take a few minutes.')
    # Shuffle the examples and collect them into batch_size batches.
    # (Internally uses a RandomShuffleQueue.)
    # We run this in two threads to avoid being a bottleneck.
    images, sparse_labels = tf.train.shuffle_batch(
        [image, label], batch_size=batch_size, num_threads=10,
        capacity=1000 + 3 * batch_size,
        # Ensures a minimum amount of shuffling of examples.
        min_after_dequeue=1000
        )

    return images, sparse_labels


##############################################
##############################################
##############################################



###### Creating the network

### input 

# reading from TFRecords
x1_image, y1_ = inputs(train = True, batch_size = batch_size, num_epochs = num_epochs)


# having the option to read from HDF5 file insted of TFRecord (for validation phase).
x1_image2 = tf.placeholder_with_default(x1_image, shape=[None, 120,120,107])
y1_2 = tf.placeholder_with_default(y1_, shape=[None, 4])
y1_2=tf.to_float(y1_2,name="convert_to_float")

# split the input data to RGB, MAG, PHASE
with tf.name_scope("split"):
  RGB, MAG, PHASE = tf.split(x1_image2, [3, 52, 52], 3)



##############################################
### RGB CNN 

CnnRGB = conv_2d(RGB, 64, 7, activation=act_fn, regularizer="L2",name="conv1_1")
tf.summary.scalar('mean', tf.reduce_mean(CnnRGB))

CnnRGB = local_response_normalization(CnnRGB,name="bn_1")
CnnRGB = conv_2d(CnnRGB, 32, 5, activation=act_fn, regularizer="L2",name="conv2_1")
CnnRGB = max_pool_2d(CnnRGB, 3,name="max_pool2_1")
CnnRGB = conv_2d(CnnRGB, 256, 3, activation=act_fn, regularizer="L2",name="conv3_1")
CnnRGB = max_pool_2d(CnnRGB, 3,name="max_pool3_1")
CnnRGB = fully_connected(CnnRGB, 512, activation=act_fn,name="FC1_1")
keep_prob1  = tf.placeholder(tf.float32)
CnnRGB = dropout(CnnRGB, keep_prob1,name="dropout1_1")
CnnRGB = fully_connected(CnnRGB, 64, activation=act_fn,name="FC2_1")
CnnRGB = dropout(CnnRGB, keep_prob1,name="dropout2_1")

# Fully connected Output layer
with tf.name_scope("y1"):
  W_fc2 = weight_variable([64, 4], 'weightCnn1fc2')
  b_fc2 = bias_variable([4], 'biasCnn1fc2')
  y1 =tf.matmul(CnnRGB, W_fc2) + b_fc2

##############################################
### MAG CNN

CnnMag = conv_2d(MAG, 64, 7, activation=act_fn, regularizer="L2",name="conv1_2")
tf.summary.scalar('mean', tf.reduce_mean(CnnMag))


CnnMag = local_response_normalization(CnnMag,name="bn_2")
CnnMag = conv_2d(CnnMag, 64, 5, activation=act_fn, regularizer="L2",name="conv2_2")
CnnMag = max_pool_2d(CnnMag, 3,name="max_pool2_2")
CnnMag = conv_2d(CnnMag, 256, 3, activation=act_fn, regularizer="L2",name="conv3_2")
CnnMag = max_pool_2d(CnnMag, 3,name="max_pool3_2")
CnnMag = fully_connected(CnnMag, 512, activation= act_fn,name="FC1_2")
keep_prob2  = tf.placeholder(tf.float32)
CnnMag = dropout(CnnMag, keep_prob2,name="dropout1_2")
CnnMag = fully_connected(CnnMag, 64, activation=act_fn,name="FC2_2")
CnnMag = dropout(CnnMag, keep_prob2,name="dropout2_2")

# Fully connected Output layer
with tf.name_scope("y2"):
  W2_fc2 = weight_variable([64, 4], 'weightCnn2fc2')
  b2_fc2 = bias_variable([4], 'biasCnn2fc2')
  y2 = tf.matmul(CnnMag, W2_fc2) + b2_fc2

##############################################
### CNN PHASE

CnnPhase = conv_2d(PHASE, 64, 7, activation=act_fn, regularizer="L2",name="conv1_3")
tf.summary.scalar('mean', tf.reduce_mean(CnnPhase))

CnnPhase = local_response_normalization(CnnPhase,name="bn_3")
CnnPhase = conv_2d(CnnPhase, 32, 5, activation=act_fn, regularizer="L2",name="conv2_3")
CnnPhase = max_pool_2d(CnnPhase, 3,name="max_pool2_3")
CnnPhase = conv_2d(CnnPhase, 256, 3, activation=act_fn, regularizer="L2",name="conv3_3")
CnnPhase = max_pool_2d(CnnPhase, 3,name="max_pool3_3")
CnnPhase = fully_connected(CnnPhase, 512, activation=act_fn,name="FC1_3")
keep_prob3  = tf.placeholder(tf.float32)

CnnPhase = dropout(CnnPhase, keep_prob3,name="dropout1_3")
CnnPhase = fully_connected(CnnPhase, 64, activation=act_fn,name="FC2_3")

CnnPhase = dropout(CnnPhase, keep_prob3,name="dropout2_3")

# Fully connected Output layer
with tf.name_scope("y3"):
  W3_fc2 = weight_variable([64, 4], 'weightCnn3fc3')
  b3_fc2 = bias_variable([4], 'biasCnn3fc3')
  y3 = tf.matmul(CnnPhase, W3_fc2) + b3_fc2

##############################################
###  cross-entropy 

with tf.name_scope("cross_entropy"):
  # defining weights
  wFuse1 = tf.Variable(1, dtype=tf.float32, trainable=True, name='weightFuse1')
  wFuse2 = tf.Variable(1, dtype=tf.float32, trainable=True, name='weightFUse2')
  wFuse3 = tf.Variable(1, dtype=tf.float32, trainable=True, name='weightFUse3')

  # Normalizing weights
  wFuse1_relative=wFuse1/(wFuse1+wFuse2+wFuse3)
  wFuse2_relative=wFuse2/(wFuse1+wFuse2+wFuse3)
  wFuse3_relative=wFuse3/(wFuse1+wFuse2+wFuse3)

  # writing the summaries
  tf.summary.scalar('wFuse1',wFuse1_relative)
  tf.summary.scalar('wFuse2',wFuse2_relative)
  tf.summary.scalar('wFuse3',wFuse3_relative)

  # multiplying weights
  y1_weighted = wFuse1_relative*y1
  y2_weighted = wFuse2_relative*y2
  y3_weighted = wFuse3_relative*y3
  y_weighted = (y1_weighted + y2_weighted + y3_weighted )/3

  # applying softmax (so that sum of the probabilities are 1)
  y_weighted_softmax=tf.nn.softmax(y_weighted,name="softmax")

  # cross-entropy 
  CrossEntropy3_weighted = tf.reduce_mean(-tf.reduce_sum(y1_2 * tf.log(y_weighted_softmax+0.00000001), reduction_indices=[1]))

  cross_entropy_weighted=CrossEntropy3_weighted #+ added_loss1 + added_loss2
  tf.summary.scalar('cost',cross_entropy_weighted)


###  calculating accuracy
with tf.name_scope("accuracy"):
  correct_prediction = tf.equal(tf.argmax(y_weighted_softmax, 1), tf.argmax(y1_2, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name='accuracy')
  tf.summary.scalar('accuracy', accuracy)

###  training step
with tf.name_scope("train"):
  train_step = tf.train.AdamOptimizer(lr).minimize(cross_entropy_weighted)

##############################################
##############################################
##############################################

with tf.Session() as sess:
  merged = tf.summary.merge_all()
  #  writer=tf.summary.FileWriter("mnist_demo/")

  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()

  # creating variable to write summaries
  train_writer = tf.summary.FileWriter("summaries" + '/train', sess.graph)
  test_writer = tf.summary.FileWriter("summaries" + '/test')
  
  # Initializing the variables
  init = tf.group(tf.global_variables_initializer(),
                      tf.local_variables_initializer())
  sess.run(init)#,feed_dict={train_input: False})

  # # Start input enqueue threads.
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  done_once = False
  try:
      step = 0
      epochs_so_far = 0
      start_time = time.time()
      while not coord.should_stop():


        run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
        run_metadata = tf.RunMetadata()

        # running train step
        summary,loss_train,_=sess.run([merged,cross_entropy_weighted,train_step],feed_dict={keep_prob1: 0.5,keep_prob2: 0.5,keep_prob3: 0.5})

        # writing summaries for training phase
        train_writer.add_run_metadata(run_metadata, 'step%03d' % step)
        train_writer.add_summary(summary, step)

        # printing the results every one in a while
        if ((step) % 20) == 0  and step != 0:
          loss,acc,w1,w2,w3=sess.run([cross_entropy_weighted,accuracy,wFuse1_relative,wFuse2_relative,wFuse3_relative],feed_dict={keep_prob1: 0.5,keep_prob2: 0.5,keep_prob3: 0.5})
          print("Iter " + str(step*batch_size) + ", steps= " + str(step) +", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Minibatch Accuracy= " + \
                      "{:.5f}".format(acc) + " w1: "+str(w1) + " w2: "+str(w2) + " w3: "+str(w3))
          step += 1

        ### validation phase
        if (step >= 4500 and (not done_once)) :
          sum_acc = 0
          sum_loss = 0
          # calculating training time
          duration = time.time() - start_time
          print("training time: " + str(duration))

          print("Validation...")
          # openning file to write the validation results
          f = open("test_set.txt", "w")

          # going over half of the test set
          for i in range((num_iter/2)-2):
            # reading next bach
            x_temp,y_temp = ob.get_next_batch()
            # feeding the batch to the network
            summary,loss,acc,pred = sess.run([merged,cross_entropy_weighted,accuracy,y_weighted_softmax],feed_dict={x1_image2: x_temp,y1_2:y_temp,keep_prob1: 1,keep_prob2:1,keep_prob3: 1})
            print(i,loss,acc)
            # writing the summaries into test summary
            test_writer.add_summary(summary, step)
            # calculating average of loss and acc so far
            sum_acc  += acc
            sum_loss += loss
            print(sum_acc/(i+1))
            print(sum_loss/(i+1))

            # if acc < 90%, then write the predicted lable and ground-truth into file
            if (acc < 0.90):
              for prediction, ground_truth in zip(np.argmax(pred,axis=1),np.argmax(y_temp,axis=1)):
                f.write("%s %s\n" % (prediction,ground_truth))

          f.close()

          print ("Validation acc: " + str(sum_acc/(i+1)) + " Validation loss: " + str(sum_loss/(i+1)) )
          sum_acc = 0
          sum_loss = 0
          f = open("test_set2.txt", "w")
          # feeding the next half of test set
          for i in range((num_iter/2)):
            x_temp,y_temp = ob2.get_next_batch()
            # feeding the batch to the network
            summary,loss,acc = sess.run([merged,cross_entropy_weighted,accuracy],feed_dict={x1_image2: x_temp,y1_2:y_temp,keep_prob1: 1,keep_prob2:1,keep_prob3: 1})
            print(i,loss,acc)
            # writing the summaries into test summary
            test_writer.add_summary(summary, step)
            # calculating average of loss and acc so far
            sum_acc  += acc
            sum_loss += loss
            print(sum_acc/(i+1))
            print(sum_loss/(i+1))

            # if acc < 90%, then write the predicted lable and ground-truth into file
            if (acc < 0.90):
              for prediction, ground_truth in zip(np.argmax(pred,axis=1),np.argmax(y_temp,axis=1)):
                f.write("%s %s\n" % (prediction,ground_truth))
          f.close()

          print ("Validation acc: " + str(sum_acc/(i+1)) + " Validation loss: " + str(sum_loss/(i+1)) )
          # save the network
          saver.save(sess, "Snapshots/model", global_step=step)
          print("saved.")
          done_once = True
        
        step += 1
        if (step* batch_size % num_train_samples == 0 and step != 0):
          epochs_so_far +=1

  except tf.errors.OutOfRangeError:
      saver.save(sess, "Snapshots/model", global_step=step)
      print('Done training for %d epochs, %d steps.' % (num_epochs, step))
  finally:
    #   # When done, ask the threads to stop.
    coord.request_stop()
    # # Wait for threads to finish.
  coord.join(threads)

