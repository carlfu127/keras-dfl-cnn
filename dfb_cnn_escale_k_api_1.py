# -*-coding:utf-8-*-

from __future__ import print_function

from keras import backend as K
import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3

from keras.models import Model
from keras.models import Sequential

from keras.optimizers import SGD
from keras.optimizers import RMSprop
from keras.optimizers import Adam
from keras.optimizers import Adadelta
from keras.optimizers import Adagrad
from keras.optimizers import Adamax
from keras.optimizers import Nadam
from keras.optimizers import TFOptimizer


from keras.layers import Input
from keras.layers import Flatten
from keras.layers import GlobalMaxPool2D
from keras.layers import GlobalAveragePooling2D
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import AvgPool2D
from keras.layers import MaxPool2D
from keras.layers import AvgPool1D
from keras.layers import MaxPool1D
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import BatchNormalization
from keras.regularizers import l2
from keras.regularizers import l1
from keras.regularizers import l1_l2

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

from keras.metrics import sparse_categorical_accuracy
from keras.metrics import sparse_categorical_crossentropy
from keras.metrics import sparse_top_k_categorical_accuracy
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.metrics import top_k_categorical_accuracy


import h5py
import numpy as np
import os
import cv2
import time
import argparse
import copy
from random import shuffle

from keras.engine import Layer

OS = "LINUX" #LINUX


def focal_loss(y_true, y_pred):

    return 0


def ctm_loss(y_true, y_pred):
    # print("ctm_loss")
    pred_list = tf.split(y_pred, 3, axis=-1)
    # true_list = tf.split(y_true, 3, axis=-1)
    loss_list = []
    for pred in pred_list:
        loss_list.append(categorical_crossentropy(y_true, pred))
    # print(loss_list)
    # print(loss_list[0] + loss_list[1] + 0.1*loss_list[2])
    return loss_list[0] + loss_list[1] + 0.1*loss_list[2]

def ctm_acc1(y_true, y_pred):
    # print("ctm_acc1")
    # true = tf.split(y_true, 3, axis=-1)[0]
    pred_list = tf.split(y_pred, 3, axis=-1)
    pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]
    # print(categorical_accuracy(true, pred))
    return categorical_accuracy(y_true, pred)

def ctm_acck(y_true, y_pred):
    # print("ctm_acck")
    # true = tf.split(y_true, 3, axis=-1)[0]
    pred_list = tf.split(y_pred, 3, axis=-1)
    pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]
    return top_k_categorical_accuracy(y_true, pred, k=3)

# path_data = "F:\\data\\electronic_scale\\"
# path_data = "./datasets/"
# output_train = path_data + 'trainset.h5'
# output_test = path_data + 'testset.h5'

class escale_test(object):
    def __init__(self, gpu_id=5):
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        num_class = 12
        BATCH_SIZE = 32
        k = 10

        fgc_base = MobileNet(input_shape=(224, 224, 3),
                             include_top=False,
                             weights=None,
                             alpha=1.)
        fgc_base.trainable = True
        # fgc_base.summary()
        feature2 = fgc_base.get_layer("conv_pw_11_relu").output
        fc_model = Model(fgc_base.inputs[0], [fgc_base.output, feature2])

        # fc_model.summary()

        input_tensor = Input(shape=(224, 224, 3))
        input_tensor_bn = BatchNormalization()(input_tensor)
        features = fc_model(input_tensor_bn)
        fc_obj = GlobalMaxPool2D()(features[0])
        fc_obj = Dropout(0.7)(fc_obj)
        fc_obj = Dense(12, activation="softmax")(fc_obj)

        fc_part = Conv2D(filters=num_class * k,
                     kernel_size=(1, 1),
                     activation="relu")(features[1])
        fc_part = GlobalMaxPool2D()(fc_part)
        fc_part = Dropout(0.5)(fc_part)
        fc_ccp = Lambda(lambda tmp: tf.expand_dims(tmp, axis=-1))(fc_part)
        fc_ccp = AvgPool1D(pool_size=k)(fc_ccp)
        fc_ccp = Lambda(lambda tmp: tf.squeeze(tmp, [-1]))(fc_ccp)
        fc_ccp = Activation(activation="softmax")(fc_ccp)
        fc_part = Dense(12, activation="softmax")(fc_part)
        output = Concatenate(axis=-1)([fc_obj, fc_part, fc_ccp])

        self.dfb_cnn = Model(input_tensor, output)

        lr = 0.001
        clip_value = 0.01
        self.dfb_cnn.compile(optimizer=SGD(lr=lr,
                                           momentum=0.9,
                                           decay=1e-5,
                                           nesterov=True,
                                           clipvalue=clip_value),
                             loss=ctm_loss,
                             metrics=[ctm_acc1, ctm_acck])
        path_prefix = "./datasets/model/escale/focal_loss_2_0.25/"
        # path_prefix = "./datasets/focal_loss_2_0.25/"
        self.dfb_cnn.load_weights(filepath=path_prefix + "weights.h5", skip_mismatch=True) ######

    def test_api(self, image, topk=3, cvt=True):
        # img = cv2.imread(image)
        # img = cv2.resize(image, (224, 224), interpolation=cv2.INTER_CUBIC)
        # if cvt:
        #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = image[np.newaxis, :]
        y_pred = self.dfb_cnn.predict(img)
        pred_list = np.split(y_pred, 3, axis=-1)
        pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]
        # print(np.argmax(pred))
        index = np.argsort(pred, axis=-1)[:, ::-1]
        index = index[0][:topk]
        conf = pred[0][index]

        return index, conf
        # return np.argmax(pred, axis=-1)

