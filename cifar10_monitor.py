from tensorflow.contrib.seq2seq.python.ops import loss

__author__ = 'kevin'
from pyimagesearch.callbacks.trainingmonitor import TrainingMonitor
from sklearn.preprocessing import LabelBinarizer
from pyimagesearch.nn.conv.minivvgnet import MiniVGGNet
from keras.optimizers import SGD
from keras.datasets import cifar10
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-o", "--output", required=True, help="path to output directory")
args = vars(ap.parse_args())

print("[INFO] process ID: {}".format(os.getpid()))

((trainX, trainY), (testX, testY)) = cifar10.load_data()
lb = LabelBinarizer()

trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

# initialize the label names for the CIFAR-10 dataset
labelNames = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]

print("[INFO] compiling model... ")
opt = SGD(lr=0.01, momentum=0.9, nesterov=True)

model = MiniVGGNet.built(width=32, height=32, depth=3, classes=10)

model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

figPath = os.path.sep.join([args["output"], "{}.jpg".format(os.getpid())])

jsonPath = os.path.sep.join([args["output"], "{}.json".format(os.getpid())])

callbacks = [TrainingMonitor(figPath, jsonPath=jsonPath)]

print("[INFO] training network...")

model.fit(trainX, trainY, validation_data=(trainX, trainY), batch_size=64, epochs=100, callbacks=callbacks, verbose=1)
