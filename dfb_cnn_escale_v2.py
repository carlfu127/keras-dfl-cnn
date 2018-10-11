# -*-coding:utf-8-*-

from __future__ import print_function

from keras import backend as K
import tensorflow as tf

from keras.applications.vgg19 import VGG19
from keras.applications.vgg16 import VGG16
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenetv2 import MobileNetV2
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
from keras.layers import GlobalMaxPool2D
from keras.layers import Lambda
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Conv2D
from keras.layers import BatchNormalization
from keras.layers import AvgPool1D
from keras.layers import Concatenate
from keras.layers import Activation

from keras.callbacks import ModelCheckpoint
from keras.callbacks import CSVLogger
from keras.callbacks import TensorBoard
from keras.callbacks import EarlyStopping

from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.metrics import top_k_categorical_accuracy


import h5py
import os
import argparse

OS = 'LINUX'

def focal_loss(y_true, y_pred, gamma=2, alpha=0.25):
    alpha_factor = K.ones_like(y_true) * alpha
    alpha_factor = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
    focal_weight = tf.where(K.equal(y_true, 1), 1 - y_pred, y_pred)
    focal_weight = alpha_factor * focal_weight ** gamma
    cls_loss = focal_weight * K.binary_crossentropy(y_true, y_pred)
    return tf.reduce_sum(cls_loss)


def ctm_loss(y_true, y_pred):
    pred_list = tf.split(y_pred, 3, axis=-1)
    loss_list = []
    for pred in pred_list:
        loss_list.append(categorical_crossentropy(y_true, pred))
    return loss_list[0] + loss_list[1] + 0.1*loss_list[2]

def ctm_acc1(y_true, y_pred):
    pred_list = tf.split(y_pred, 3, axis=-1)
    pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]
    return categorical_accuracy(y_true, pred)

def ctm_acck(y_true, y_pred):
    pred_list = tf.split(y_pred, 3, axis=-1)
    pred = pred_list[0] + pred_list[1] + 0.1 * pred_list[2]
    return top_k_categorical_accuracy(y_true, pred, k=3)

def create_model(input_shape, num_class, k):
    fgc_base = MobileNetV2(input_shape=input_shape,
                           include_top=False,
                           weights=None,
                           alpha=1.)
    fgc_base.trainable = True
    # fgc_base.summary()
    feature2 = fgc_base.get_layer("block_11_expand_relu").output
    fc_model = Model(fgc_base.inputs[0], [fgc_base.output, feature2])

    fc_model.summary()

    input_tensor = Input(shape=input_shape)
    input_tensor_bn = BatchNormalization()(input_tensor)
    features = fc_model(input_tensor_bn)
    fc_obj = GlobalMaxPool2D()(features[0])
    fc_obj = Dropout(0.7)(fc_obj)
    fc_obj = Dense(num_class, activation="softmax")(fc_obj)

    fc_part = Conv2D(filters=num_class * k,
                     kernel_size=(1, 1),
                     activation="relu")(features[1])
    fc_part = GlobalMaxPool2D()(fc_part)
    fc_part = Dropout(0.5)(fc_part)
    fc_ccp = Lambda(lambda tmp: tf.expand_dims(tmp, axis=-1))(fc_part)
    fc_ccp = AvgPool1D(pool_size=k)(fc_ccp)
    fc_ccp = Lambda(lambda tmp: tf.squeeze(tmp, [-1]))(fc_ccp)
    fc_ccp = Activation(activation="softmax")(fc_ccp)
    fc_part = Dense(num_class, activation="softmax")(fc_part)
    output = Concatenate(axis=-1)([fc_obj, fc_part, fc_ccp])

    return Model(input_tensor, output)

def create_callbacks(path_prefix)  :

    csvlog = CSVLogger(filename=path_prefix + "log.csv",
                       separator=' ',
                       append=True)
    ckpt = ModelCheckpoint(filepath=path_prefix + "weights.h5",
                           monitor='val_ctm_acc1',
                           verbose=1,
                           save_best_only=True,
                           save_weights_only=True,
                           mode='max',
                           period=1)
    tsb = TensorBoard(log_dir=path_prefix + "tensorboard/",
                      histogram_freq=5,
                      write_graph=True,
                      write_images=False)
    est = EarlyStopping(monitor='val_loss',
                        patience=3,
                        verbose=1,
                        mode='min')
    return [csvlog, ckpt, tsb, est]

def _main_(args):
    num_class = args.nc
    k = args.k
    BATCH_SIZE = args.nb
    input_shape = (224, 224, 3)
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print("Staring......")
    print("\tCreate Models......")
    model = create_model(input_shape, num_class, k)

    print("\tCompiling the model......")
    path_prefix = args.path
    call = create_callbacks(path_prefix)
    print("\tLoading Data......")

    output_train = args.train
    output_test = args.test
    with h5py.File(output_train, "r") as f:
        X_train = f["X"][:]
        Y_train = f["Y"][:]
        print(X_train.shape)
        print(Y_train.shape)
        f.close()
        print("\tLoaded Train Data......")
    with h5py.File(output_test, "r") as f:
        X_test = f["X"][:]
        Y_test = f["Y"][:]
        print(X_test.shape)
        print(Y_test.shape)
        f.close()
        print("\tLoaded Test Data......")

    last_run = 0
    lr = 0.001
    clip_value = 0.01
    print("\tTraining......")
    for i in range(5):
        print("\t...Stage %i" % (i + 1))
        model.compile(optimizer=RMSprop(lr=lr),
                        loss=ctm_loss,
                        metrics=[ctm_acc1, ctm_acck])
        hist = model.fit(x=X_train, y=Y_train,
                           batch_size=BATCH_SIZE,
                           epochs=last_run + 200,
                           verbose=1,
                           validation_data=[X_test, Y_test],
                           shuffle=True,
                           callbacks=call,
                           initial_epoch=last_run)

        print("Run", len(hist.history["loss"]), "Epochs")
        last_run += len(hist.history["loss"])
        lr *= 0.1
        clip_value *= 0.1

    print("End")

if __name__ == "__main__":

    argparser = argparse.ArgumentParser(description='train and evaluate dfl_cnn model on any dataset')

    argparser.add_argument('-train', help='train dataset', default='datasets/trainset.h5')
    argparser.add_argument('-test', help='test dataset', default='datasets/testset.h5')
    argparser.add_argument('-path', help='the path to save the model and logs', default='model/dfl_cnn/')

    argparser.add_argument('-nc' , help='number of classes', default=12)
    argparser.add_argument('-k' , help='number of patches', default=10)
    argparser.add_argument('-nb' , help='number of classes', default=32)

    args = argparser.parse_args()
    _main_(args)