from tensorflow.python.data.experimental.ops.optimization import model

__author__ = 'kevin'
from keras.applications import VGG16
from keras.applications import imagenet_utils
from keras.preprocessing.image import img_to_array, load_img
from sklearn.preprocessing import LabelEncoder
from pyimagesearch.io.hdf5datasetwriter import HDF5DatasetWriter
from imutils import paths
import numpy as np
import argparse
import os
import random
import progressbar

ap = argparse.ArgumentParser()

ap.add_argument("-d", "--dataset", help="path to input dataset", required=True)
ap.add_argument("-o", "--output", help="path to output HDF5 file", required=True)
ap.add_argument("-b", "--batchsize", help="batch size of image to be passed through network", type=int, default=32)
ap.add_argument("-s", "--buffersize", help="size of feature extraction buffer", default=1000, type=int)
args = vars(ap.parse_args())

bs = args["batchsize"]

print("[INFO] loading images... ")

imagePaths = list(paths.list_images(args["dataset"]))

random.shuffle(imagePaths)

labels = [p.split(os.path.sep)[-2] for p in imagePaths]
le = LabelEncoder()
labels = le.fit_transform(labels)

print("[INFO] loading network...")
model = VGG16(weights="imagenet", include_top=False)

dataset = HDF5DatasetWriter((len(imagePaths), 512 * 7 * 7), args["output"], dataKey="features",
                            bufSize=args["buffersize"])
dataset.storeClassLables(le.classes_)

widgets = ["Extracting Features: ", progressbar.Percentage(), " ", progressbar.Bar(), " ", progressbar.ETA()]

pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()

for i in np.arange(0, len(imagePaths), bs):
    batchPaths = imagePaths[i:i + bs]
    batchLabel = labels[i:i + bs]
    batchImages = []

    for (j, imagePath) in enumerate(batchPaths):
        image = load_img(imagePath, target_size=(224, 224))
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = imagenet_utils.preprocess_input(image)

        batchImages.append(image)

    batchImages = np.vstack(batchImages)
    features = model.predict(batchImages, batch_size=bs)
    features = features.reshape((features.shape[0], 512 * 7 * 7))
    dataset.add(features, batchLabel)
    pbar.update(i)

dataset.close()
pbar.finish()
