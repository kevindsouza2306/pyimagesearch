__author__ = 'kevin'
from keras.layers.core import Dropout, Flatten, Dense


class FCHeadNet:
    @staticmethod
    def built(baseModel, classes, D):
        headModel = baseModel.output
        headModel = Flatten(name="flatten")(headModel)
        headModel = Dense(D, activation="relu")(headModel)
        headModel = Dropout(0.5)(headModel)
        # add a softmax layer
        headModel = Dense(classes, activation="softmax")(headModel)

        return headModel
