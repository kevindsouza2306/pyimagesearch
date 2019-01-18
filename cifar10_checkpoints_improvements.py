from pyimagesearch.nn.conv.minivvgnet import MiniVGGNet
from sklearn.preprocessing import LabelBinarizer
from keras.datasets import cifar10
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint
import os
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-w", "--weights", required=True, help="Path to weight directory")

args = vars(ap.parse_args())

print("[INFO] Loading datasets......")

((trainX, trainY), (testX, testY)) = cifar10.load_data()

trainX = trainX.astype("float") / 255.0
testX = testX.astype("float") / 255.0

trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

print("[INFO] Compiling model...")

opt = SGD(lr=0.01, decay=0.01 / 40, momentum=0.9, nesterov=True)

model = MiniVGGNet.built(width=32, height=32, depth=3, classes=10)
model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])

fname = os.path.sep.join([args["weights"], "weights-{epoch:03d}-{val_loss:.4f}.hdf5"])

checkpoint = ModelCheckpoint(fname, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
callbacks = [checkpoint]

print("[INFO] Training network")

model.fit(trainX, trainY, validation_data=(testX, testY), batch_size=64, epochs=40, callbacks=callbacks, verbose=2)

