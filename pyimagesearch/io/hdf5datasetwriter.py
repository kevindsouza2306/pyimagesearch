from numpy import unicode

__author__ = 'kevin'

import h5py
import os


class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey="images", bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError(
                "The supplied 'output' file alredy exists and cannot be overwritten. Manually delete the file before continuing",
                outputPath)
        self.db = h5py.File(outputPath, "w")
        self.data = self.db.create_dataset(dataKey, dims, dtype="float")
        self.lables = self.db.create_dataset("labels", (dims[0],), dtype="int")
        self.bufSize = bufSize
        self.buffer = {"data": [], "labels": []}
        self.idx = 0

    def add(self, rows, lables):
        self.buffer["data"].extend(rows)
        self.buffer["labels"].extend(lables)

        if len(self.buffer["data"]) >= self.bufSize:
            self.flush()

    def flush(self):

        i = self.idx + len(self.buffer["data"])
        self.data[self.idx:i] = self.buffer["data"]
        self.lables[self.idx:i] = self.buffer["labels"]
        self.idx = i
        self.buffer = {"data": [], "labels": []}

    def storeClassLables(self, classLables):
        dt = h5py.special_dtype(vlen=unicode)
        labelSet = self.db.create_dataset("label_names", (len(classLables),), dtype=dt)
        labelSet[:] = classLables

    def close(self):

        if len(self.buffer["data"]) > 0:
            self.flush


        self.db.close()
