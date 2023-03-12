
import numpy as np
import cv2
import glob, os
import tensorflow as tf
from imgaug import augmenters as iaa

resize_size = 180

os.chdir("C:/Users/Downloads/Deep learning/MaskContamination/TF231/version5/data/train/clean")
path = 'C:/Users/Downloads/Deep learning/MaskContamination/TF231/version5/data/train/augmented/clean/'
i = 0
files = glob.glob('*.jpg')
# files.extend(glob.glob('*.JPG'))
# files.extend(glob.glob('*.png'))

for file in files: #glob.glob("*.jpg", "*.JPG"):
	i = i + 1
	print(file)
	image = cv2.imread(file)
	resized_image = cv2.resize(image, (resize_size, resize_size)) 
	resized_image.shape
	imageName = "%d_1.jpg" % i
	# save original image
	cv2.imwrite(os.path.join(path ,imageName), resized_image)

	# blurer1 = iaa.GaussianBlur(0.5)
	# imageGaussian = blurer1.augment_image(resized_image) 
	# # save GaussianBlur image
	# imageName = "%d_2.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageGaussian)

	# adder2 = iaa.Add(10)
	# imageAdded = adder2.augment_image(resized_image) 
	# # save Add image
	# imageName = "%d_2.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageAdded)

	# adder3 = iaa.Add(25)
	# imageAdded = adder3.augment_image(resized_image) 
	# # save Add image
	# imageName = "%d_4.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageAdded)

	flipper = iaa.Fliplr(1.0)
	imageFlipped = flipper.augment_image(resized_image)
	# save flipped image
	imageName = "%d_2.jpg" % i
	cv2.imwrite(os.path.join(path ,imageName), imageFlipped)

	# bilateralBlur1 = iaa.BilateralBlur(3)
	# imageBilateralBlur = bilateralBlur1.augment_image(resized_image) 
	# # save BilateralBlur image
	# imageName = "%d_6.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageBilateralBlur)

	# contrast2 = iaa.ContrastNormalization(1.2)
	# imageContrast = contrast2.augment_image(resized_image) 
	# # save ContrastNormalization image
	# imageName = "%d_4.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageContrast)

	# gray1 = iaa.Grayscale(alpha=0.5)
	# imageGray = gray1.augment_image(resized_image) 
	# # save Grayscale image
	# imageName = "%d_5.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageGray)

	# medianBlur1 = iaa.MedianBlur(k=3)
	# imageMedian = medianBlur1.augment_image(resized_image) 
	# # save MedianBlur image
	# imageName = "%d_9.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageMedian)

	# emboss1 = iaa.Emboss(alpha=1, strength=0)
	# imageEmboss = emboss1.augment_image(resized_image) 
	# # save Emboss image
	# imageName = "%d_6.jpg" % i
	# cv2.imwrite(os.path.join(path ,imageName), imageEmboss)
