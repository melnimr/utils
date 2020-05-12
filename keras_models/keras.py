# cnn model for the cifar10 problem with test-time augmentation
import numpy
from numpy import argmax
from numpy import mean
from numpy import std
from numpy import expand_dims
from sklearn.metrics import accuracy_score
from keras.datasets.cifar10 import load_data
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import BatchNormalization

# load and return the cifar10 dataset ready for modeling
def load_dataset():
	# load dataset
	(trainX, trainY), (testX, testY) = load_data()
	# normalize pixel values
	trainX = trainX.astype('float32') / 255
	testX = testX.astype('float32') / 255
	# one hot encode target values
	trainY = to_categorical(trainY)
	testY = to_categorical(testY)
	return trainX, trainY, testX, testY

# define the cnn model for the cifar10 dataset
def define_model():
	# define model
	model = Sequential()
	model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform', input_shape=(32, 32, 3)))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(MaxPooling2D((2, 2)))
	model.add(Flatten())
	model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
	model.add(BatchNormalization())
	model.add(Dense(10, activation='softmax'))
	# compile model
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	return model

# make a prediction using test-time augmentation
def tta_prediction(datagen, model, image, n_examples):
	# convert image into dataset
	samples = expand_dims(image, 0)
	# prepare iterator
	it = datagen.flow(samples, batch_size=n_examples)
	# make predictions for each augmented image
	yhats = model.predict_generator(it, steps=n_examples, verbose=0)
	# sum across predictions
	summed = numpy.sum(yhats, axis=0)
	# argmax across classes
	return argmax(summed)

# evaluate a model on a dataset using test-time augmentation
def tta_evaluate_model(model, testX, testY):
	# configure image data augmentation
	datagen = ImageDataGenerator(horizontal_flip=True)
	# define the number of augmented images to generate per test set image
	n_examples_per_image = 7
	yhats = list()
	for i in range(len(testX)):
		# make augmented prediction
		yhat = tta_prediction(datagen, model, testX[i], n_examples_per_image)
		# store for evaluation
		yhats.append(yhat)
	# calculate accuracy
	testY_labels = argmax(testY, axis=1)
	acc = accuracy_score(testY_labels, yhats)
	return acc

# fit and evaluate a defined model
def evaluate_model(model, trainX, trainY, testX, testY):
	# fit model
	model.fit(trainX, trainY, epochs=3, batch_size=128, verbose=0)
	# evaluate model using tta
	acc = tta_evaluate_model(model, testX, testY)
	return acc

# repeatedly evaluate model, return distribution of scores
def repeated_evaluation(trainX, trainY, testX, testY, repeats=10):
	scores = list()
	for _ in range(repeats):
		# define model
		model = define_model()
		# fit and evaluate model
		accuracy = evaluate_model(model, trainX, trainY, testX, testY)
		# store score
		scores.append(accuracy)
		print('> %.3f' % accuracy)
	return scores

# load dataset
trainX, trainY, testX, testY = load_dataset()
# evaluate model
scores = repeated_evaluation(trainX, trainY, testX, testY)
# summarize result
print('Accuracy: %.3f (%.3f)' % (mean(scores), std(scores)))
