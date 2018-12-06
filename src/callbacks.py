from time import time
import numpy as np
import os

from keras.callbacks import Callback
from keras.callbacks import ModelCheckpoint

from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix

import shutil


class TimingCallback(Callback):
    def __init__(self):
        self.logs = {}

    def on_epoch_begin(self, epoch, logs={}):
        self.epoch_start = time()

    def on_epoch_end(self, epoch, logs={}):
        self.logs[epoch] = {}
        self.logs[epoch]['epoch_time'] = time() - self.epoch_start
        self.logs[epoch]['mean_batch_time'] = np.mean(self.batch_logs)

    def on_batch_begin(self, batch, logs={}):
        self.batch_logs = []
        self.batch_start = time()

    def on_batch_end(self, batch, logs={}):
        self.batch_logs.append(time() - self.batch_start)


class Metrics(Callback):
    def on_train_begin(self, logs={}):
        self.confusion = []
        self.precision = []
        self.recall = []
        self.f1s = []
        
    def on_epoch_end(self, epoch, logs={}):
        # shuffling
        idx = np.random.permutation(len(self.validation_data[0]))
        # get predictions and shuffled target
        pred = self.model.predict(self.validation_data[0][idx])[:, 0]
        pred = np.round(pred)
        targ = self.validation_data[1][idx, 0]

        self.confusion.append(confusion_matrix(targ, pred))
        self.precision.append(precision_score(targ, pred))
        self.recall.append(recall_score(targ, pred))
        self.f1s.append(f1_score(targ, pred))

        print("\nPrecision:", self.precision[-1])
        print("\nRecall:", self.recall[-1])
        print("\nConfusion matrix:")
        print(self.confusion[-1])
        print("\nF1 score =", self.f1s[-1])
        print()


# # create callbacks
# time_cb = TimingCallback()
# metrics_cb = Metrics()

# __filepath = "b-best-model-{epoch:02d}-{val_loss:.4f}.hdf5"
# checkpoint = ModelCheckpoint(os.path.join(__model_dir, __filepath), 
#     mode='min', save_best_only=True, verbose=1)


