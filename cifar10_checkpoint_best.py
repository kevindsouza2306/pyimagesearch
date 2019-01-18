from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
from pyimagesearch.nn.conv.minivvgnet import MiniVGGNet
import argparse
import os

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", help="path to best weights files")
args = vars(ap.parse_args())

print("[INFO] Loading CIFAR-10 dataset")
((trainX, trainY), (testX, testY)) = cifar10.load_data()
trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

lb = LabelBinarizer()
trainY = lb.fit_transform(trainY)
testY = lb.fit_transform(testY)

print("[INFO] Compiling model....")

opt = SGD(lr=0.01, nesterov=True, momentum=0.9, decay=0.01 / 40)

model = MiniVGGNet.built(width=32, height=32, depth=3, classes=10)

model.compile(loss="categorical_crossentopy", optimizer=opt, metrics=["accuracy"])

checkpoints = ModelCheckpoint(args["weights"], monitor="val_loss", save_best_only=True, verbose=1)

callback = [checkpoints]
print("[INFO] Training Network")
model.fit(trainX, trainY, validation_data=(testX, testY), callbacks=callback, epochs=40, verbose=2)
