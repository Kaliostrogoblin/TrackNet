import tensorflow as tf
import math

def accuracy(y_true, y_pred):
    acc = tf.equal(y_true[:, 0], tf.round(y_pred[:, 0]))
    return tf.keras.backend.mean(acc, axis=-1)


def circle_area(y_true, y_pred):
    def area(R1, R2):
        return R1*R2*math.pi
    areas = area(y_pred[:, 3], y_pred[:, 4])
    return areas


def point_in_square(y_true, y_pred):
    # checks if the next point of seed 
    # is included in predicted square
    # 1 - if yes, 0 - otherwise
    # x coords
    x_coord_true = y_true[:, 0]
    x_coord_pred = y_pred[:, 0]
    # y coords
    y_coord_true = y_true[:, 1]
    y_coord_pred = y_pred[:, 1]
    # coordinates distances
    x_dist = tf.abs(x_coord_true - x_coord_pred)
    y_dist = tf.abs(y_coord_true - y_coord_pred)
    # check if distances smaller than semiaxis
    xsemiaxis = tf.greater_equal(y_pred[:, 2], x_dist)
    ysemiaxis = tf.greater_equal(y_pred[:, 3], y_dist)
    # return True if point in cicrle
    return tf.logical_and(xsemiaxis, ysemiaxis)


def point_in_ellipse(y_true, y_pred):
    # checks if the next point of seed 
    # is included in predicted circle
    # 1 - if yes, 0 - otherwise
    # x coords
    x_coord_true = y_true[:, 0]
    x_coord_pred = y_pred[:, 0]
    # y coords
    y_coord_true = y_true[:, 1]
    y_coord_pred = y_pred[:, 1]
    # coordinate's distances
    x_dist = tf.square(x_coord_pred - x_coord_true)
    y_dist = tf.square(y_coord_pred - y_coord_true)
    # calculate x and y parts of equation
    x_part = x_dist / tf.square(y_pred[:, 2])
    y_part = y_dist / tf.square(y_pred[:, 3])
    # left size of equation x_part + y_part = 1
    left_size = x_part + y_part
    # if left size less than 1, than point in ellipse
    return tf.less_equal(left_size, 1)


def point_in_area(y_true, y_pred):
    # gather rows with real tracks
    cond = tf.equal(y_true[:, 0], 1) # condition
    indices = tf.where(cond) # indices of real tracks
    # gathering
    # output shape: (?, 1, ?)
    y_pred_t = tf.gather(y_pred, indices)
    y_true_t = tf.gather(y_true, indices)
    # squeezing tensors and exclude probabilities
    y_pred_t = tf.squeeze(y_pred_t, axis=1)[:, 1:]
    y_true_t = tf.squeeze(y_true_t, axis=1)[:, 1:]
    # calculate loss excluding probability
    return point_in_ellipse(y_true_t, y_pred_t)