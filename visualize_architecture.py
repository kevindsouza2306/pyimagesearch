__author__ = 'kevin'
from pyimagesearch.nn.conv.lenet import LeNet
from keras.utils import plot_model

model = LeNet.built(28, 28, 1, 10)

plot_model(model, to_file="lenet.png", show_shapes=True)
