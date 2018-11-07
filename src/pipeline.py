import numpy as np
import keras.backend as K
import tensorflow as tf
from time import time
from metrics import point_in_ellipse


def timeit(func):
    def timed(*args, **kw):
        ts = time()
        result = func(*args, **kw)
        te = time()
        # print exec time
        print('%r %2.2f ms' % (func.__name__, (te - ts) * 1000))
        return result
    return timed


def create_arrays(n=1000, m=500):
    target = np.random.randint(50, size=(n, 2))
    preds = np.random.randint(50, size=(m, 4))
    return target, preds


@timeit
def get_potential_tracks_idx(preds, threshold = 0.5):
    return tf.where(preds[:, 0] > threshold)


@timeit
def get_points_in_ellipses_mask(hits, ellipses):
    f = lambda x: point_in_ellipse(hits, tf.expand_dims(x, 0))
    return tf.map_fn(f, ellipses, dtype=tf.bool)


@timeit
def expand_inputs(inputs, hits, mask):
    idx = tf.where(mask)
    f = lambda x: tf.concat([inputs[x[0]], tf.expand_dims(hits[x[1]], 0)], axis=0)
    return tf.map_fn(f, idx, dtype=tf.float32)


@timeit
def catch_and_prolong(preds, inputs, hits):
    if len(preds.shape) == 5:
        # 5 means that preds[:, 0] - probability
        tracks_idx = get_potential_tracks_idx(preds)
        # remove from inputs non tracks
        inputs = inputs[tracks_idx]
        # drop probabilities from tensor
        ellipses = preds[tracks_idx, 1:]
    else:
        ellipses = preds
    # pie_mask - point in ellipse mask
    pie_mask = get_points_in_ellipses_mask(hits, ellipses)
    breaked_tracks = inputs[tf.logical_not(tf.reduce_any(pie_mask, axis=1))]
    return expand_inputs(inputs, hits, pie_mask), breaked_tracks
    

if __name__ == '__main__':
    pass