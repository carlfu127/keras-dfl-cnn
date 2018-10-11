# -*-coding:utf-8-*-

from __future__ import print_function

from keras import backend as K
import tensorflow as tf
from keras.losses import categorical_crossentropy


OS = 'LINUX'


def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(cls_loss)

    """
    # scale preds so that the class probas of each sample sum to 1
    y_pred /= tf.reduce_sum(y_pred, len(y_pred.get_shape()) - 1, True)
    # manual computation of crossentropy
    _epsilon = tf.convert_to_tensor(1e-7, y_pred.dtype.base_dtype)
    y_pred = tf.clip_by_value(y_pred, _epsilon, 1. - _epsilon)
    gamma_ = gamma * tf.ones_like(y_pred)
    return -tf.reduce_sum(alpha * tf.pow(1-y_pred, gamma_) * y_true * tf.log(y_pred),
                           len(output.get_shape()) - 1)

    """
def pc_loss(y_true, y_pred):
    alpha = tf.constant(0.)
    d_preds = tf.split(y_pred, 2)
    d_trues = tf.split(y_true, 2)
    ce_loss = tf.constant(0.)
    for true, pred in zip(d_trues, d_preds):
        ce_loss += categorical_crossentropy(true, pred)
    ce_loss = K.mean(ce_loss, axis=-1)
    lamda = tf.constant(1.0)
    gamma = 1 - tf.reduce_sum(tf.multiply(d_trues[0], d_trues[1]), axis=-1)
    d_ec = tf.reduce_mean(tf.square(d_preds[0] - d_preds[1]), axis=-1)

    ec_loss = lamda * gamma * d_ec
    ec_loss = tf.reduce_mean(ec_loss, axis=-1)
    return ce_loss + alpha * ec_loss
