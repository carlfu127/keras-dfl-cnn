# ssh://ybx9191@172.17.11.82:22/usr/bin/python
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

OS = "WINDOWS" #LINUX


def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(cls_loss)



def ctm_loss(y_true, y_pred):
    # print("ctm_loss")
    pred_list = tf.split(y_pred, 3, axis=-1)
    # true_list = tf.split(y_true, 3, axis=-1)
    loss_list = []
    for pred in pred_list:
        loss_list.append(focal_loss(y_true, pred))
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
    print("ctm_acck")
    # true = tf.split(y_true, 3, axis=-1)[0]
    pred_list = tf.split(y_pred, 3, axis=-1)
    pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]
    return top_k_categorical_accuracy(y_true, pred, k=3)

# path_data = "F:\\data\\electronic_scale\\"
path_data = "./datasets/"
output_train = path_data + 'trainset.h5'
output_test = path_data + 'testset.h5'

def api(image_path, gpu_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    # img = cv2.imread(image_path)
    # img = cv2.resize(img, (192, 256), interpolation=cv2.INTER_CUBIC)
    img = image_path#[np.newaxis, :]
    num_class = 12
    BATCH_SIZE = 4
    k = 10

    fgc_base = MobileNet(input_shape=(192, 256, 3),
                         include_top=False,
                         weights=None,
                         alpha=1.)
    fgc_base.trainable = True
    # fgc_base.summary()
    feature2 = fgc_base.get_layer("conv_pw_11_relu").output
    fc_model = Model(fgc_base.inputs[0], [fgc_base.output, feature2])

    fc_model.summary()

    input_tensor = Input(shape=(192, 256, 3))
    features = fc_model(input_tensor)
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

    dfb_cnn = Model(input_tensor, output)

    lr = 0.001
    clip_value = 0.01
    dfb_cnn.compile(optimizer=SGD(lr=lr,
                                  momentum=0.9,
                                  decay=1e-5,
                                  nesterov=True,
                                  clipvalue=clip_value),
                    loss=ctm_loss,
                    metrics=[ctm_acc1, ctm_acck])
    path_prefix = "./datasets/model/escale/focal_loss_2_0.25/"

    dfb_cnn.load_weights(filepath=path_prefix + "weights.h5", skip_mismatch=True) ######
    y_pred = dfb_cnn.predict(img, batch_size=BATCH_SIZE)
    pred_list = np.split(y_pred, 3, axis=-1)
    pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]

    return np.argmax(pred, axis=-1)

if __name__ == '__main__':

    listfruit = ['紫薯', '红薯', '冰糖梨', '砀山梨', '砂糖桔', '蜜桔', '澳洲血橙', '赣南橙', '红富士', '蛇果', '小土豆', '黄心土豆']
    fruit = ['地瓜', '梨', '橘子', '橙子', '苹果', '马铃薯']

    print("\tLoading Data......")
    # with h5py.File(output_train, "r") as f:
    #     X_train = f["X"][:]
    #     Y_train = f["Y"][:]
    #     print(X_train.shape)
    #     print(Y_train.shape)
    #     f.close()
    #     print("\tLoaded Train Data......")
    with h5py.File(output_test, "r") as f:
        X_test = f["X"][:]
        Y_test = f["Y"][:]
        print(X_test.shape)
        print(Y_test.shape)
        f.close()
        print("\tLoaded Test Data......")
    print("\tVerifying the Data......")

    print(np.max(Y_test), np.min(Y_test))

    dense_y_true = np.argmax(Y_test, axis=-1)
    y_pred = api(X_test, gpu_id=3)
    cnt = 0
    cnt1 = 0
    cnt2 = 0
    cnt3 = 0
    cnt4 = 0
    cnt5 = 0
    cnt6 = 0
    cnt7 = 0
    cnt8 = 0
    cnt9 = 0
    cnt10 = 0
    cnt11 = 0
    cnt12 = 0
    err1 = 0
    err2 = 0
    err3 = 0
    err4 = 0
    err5 = 0
    err6 = 0
    err7 = 0
    err8 = 0
    err9 = 0
    err10 = 0
    err11 = 0
    err12 = 0
    for i in range(X_test.shape[0]):
        if dense_y_true[i] == 0:
            cnt1 += 1
        elif dense_y_true[i] == 1:
            cnt2 += 1
        elif dense_y_true[i] == 2:
            cnt3 += 1
        elif dense_y_true[i] == 3:
            cnt4 += 1
        elif dense_y_true[i] == 4:
            cnt5 += 1
        elif dense_y_true[i] == 5:
            cnt6 += 1
        elif dense_y_true[i] == 6:
            cnt7 += 1
        elif dense_y_true[i] == 7:
            cnt8 += 1
        elif dense_y_true[i] == 8:
            cnt9 += 1
        elif dense_y_true[i] == 9:
            cnt10 += 1
        elif dense_y_true[i] == 10:
            cnt11 += 1
        else:
            cnt12 += 1
        if y_pred[i] != dense_y_true[i]:
            if dense_y_true[i] == 0:
                err1 += 1
            elif dense_y_true[i] == 1:
                err2 += 1
            elif dense_y_true[i] == 2:
                err3 += 1
            elif dense_y_true[i] == 3:
                err4 += 1
            elif dense_y_true[i] == 4:
                err5 += 1
            elif dense_y_true[i] == 5:
                err6 += 1
            elif dense_y_true[i] == 6:
                err7 += 1
            elif dense_y_true[i] == 7:
                err8 += 1
            elif dense_y_true[i] == 8:
                err9 += 1
            elif dense_y_true[i] == 9:
                err10 += 1
            elif dense_y_true[i] == 10:
                err11 += 1
            else:
                err12 += 1
            cnt += 1
        img = cv2.cvtColor(X_test[i], cv2.COLOR_RGB2BGR)
        cv2.imshow("img", np.uint8(img)),
        print(cnt, i, "Correct", listfruit[dense_y_true[i]], "Predict", listfruit[y_pred[i]])
        cv2.waitKey(0)
    print("End")
    print("incorrect numbers: %d" %(cnt))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[0], err1, cnt1))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[1], err2, cnt2))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[2], err3, cnt3))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[3], err4, cnt4))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[4], err5, cnt5))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[5], err6, cnt6))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[6], err7, cnt7))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[7], err8, cnt8))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[8], err9, cnt9))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[9], err10, cnt10))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[10], err11, cnt11))
    print("%s incorrect numbers: %d,  all : %d" % (listfruit[11], err12, cnt12))
