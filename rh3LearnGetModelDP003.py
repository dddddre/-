# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals
from tensorflow import keras
import os
import pandas as pd
import tensorflow as tf


__all__=['rh3LearnGetModelDP003', 'rh3LearnGetModelDP003Unfit']


def inception_module(input_tensor, stride=1, activation='linear',
                     use_bottleneck=True,bottleneck_size=32,
                     kernel_size=40,nb_filters=32,pool_size=3):

    if use_bottleneck and int(input_tensor.shape[-1]) > 1:
        input_inception = keras.layers.Conv1D(filters=bottleneck_size, kernel_size=1,
                                              padding='same', activation=activation, use_bias=False)(input_tensor)
    else:
        input_inception = input_tensor
    kernel_size_s = [kernel_size // (2 ** i) for i in range(3)]
    conv_list = []
    for i in range(len(kernel_size_s)):
        conv_list.append(keras.layers.Conv1D(filters=nb_filters, kernel_size=kernel_size_s[i],
                                             strides=stride, padding='same', activation=activation, use_bias=False)(input_inception))
    max_pool_1 = keras.layers.MaxPool1D(pool_size=pool_size, strides=stride, padding='same')(input_tensor)
    conv_6 = keras.layers.Conv1D(filters=nb_filters, kernel_size=1,
                                 padding='same', activation=activation, use_bias=False)(max_pool_1)
    conv_list.append(conv_6)
    x = keras.layers.Concatenate(axis=2)(conv_list)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Activation(activation='relu')(x)

    return x


def shortcut_layer(input_tensor, out_tensor):

    shortcut_y = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias=False)(input_tensor)
    shortcut_y = keras.layers.BatchNormalization()(shortcut_y)
    x = keras.layers.Add()([shortcut_y, out_tensor])
    x = keras.layers.Activation('relu')(x)
    return x


def rh3LearnGetModelDP003Unfit(time_window=200, features=40, num_class=3,
                               depth=6, use_residual=True,
                               bottleneck_size=32, kernel_size=40, nb_filters=32, pool_size=3,
                               lstm=False, lstm_units=64, l2=0.01,
                               **kwargs):
    '''
    function:该模型为InceptionTime模型，没有训练的模型
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    author:@wwj
    input:
        该模型特定参数：
        time_window=200 时间窗口大小（必须与x中的对应维度相同）
        features=40 同一时刻的特征数（必须与x中的对应维度相同）
        num_class=3 分几类
        depth=6  网络总深度，一般取3的正整数倍
        use_residual=True 是否使用使用resNet
        bottleneck_size=32 inceptionmodule中的bottleneck的filters数
        kernel_size=40 inceptionmodule中的三个窗口的上限
        nb_filters=32  inceptionmodule中的三个窗口的filters数
        pool_size=3    inceptionmodule中的maxpool的size
        lstm=False, 是否后接LSTM
        lstm_units=64, lstm的units
        l2=0.01   ，lstm的l2正则化惩罚系数

    '''

    input_layer = keras.Input(shape=[time_window, features])  # 生成输入的占位符
    x = input_layer
    input_res = input_layer

    for d in range(depth):
        x = inception_module(x, bottleneck_size=bottleneck_size,
                             kernel_size=kernel_size, nb_filters=nb_filters, pool_size=pool_size)
        if use_residual and d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x
    if lstm:
        x = keras.layers.LSTM(lstm_units,
                              kernel_regularizer=keras.regularizers.l2(l2),
                              recurrent_regularizer=keras.regularizers.l2(l2)
                              )(x)
    else:
        x = keras.layers.GlobalAveragePooling1D()(x)

    output_layer = keras.layers.Dense(num_class, activation='softmax')(x)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    #model.summary()
    #keras.utils.plot_model(model, 'DP003.png', show_shapes=True)

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return model


def rh3LearnGetModelDP003(train_x, train_y, valid_x, valid_y,
                          time_window=200, features=40, num_class=3,
                          depth=6, use_residual=True,
                          bottleneck_size=32, kernel_size=40, nb_filters=32, pool_size=3,
                          lstm=False, lstm_units=64, l2=0.01,
                          batch_size=64, patience=50, epochs=1500,
                          **kwargs):
    '''
    function:该模型为四层卷积加上LSTM的神经网络
    author:@wwj
    input:
        train_x	ndarray	训练集特征构成的三维数组，sample_num×time_window×features，第一个维度为样本量，第二个维度为时间序列，第三个维度为特征。
        train_y	ndarray	训练集标签构成的二维数组，sample_num×labels。
        valid_x	ndarray	验证集特征构成的三维数组，相关说明同train_x。
        valid_y	ndarray	验证集标签构成的二维数组，相关说明同train_y
        该模型特定参数：
        time_window=200 时间窗口大小（必须与x中的对应维度相同）
        features=40 同一时刻的特征数（必须与x中的对应维度相同）
        num_class=3 分几类
        depth=6  网络总深度，一般取3的正整数倍
        use_residual=True 是否使用使用resNet
        bottleneck_size=32 inceptionmodule中的bottleneck的filters数
        kernel_size=40 inceptionmodule中的三个窗口的上限
        nb_filters=32  inceptionmodule中的三个窗口的filters数
        pool_size=3    inceptionmodule中的maxpool的size

        lstm=False, 是否后接LSTM
        lstm_units=64, lstm的units
        l2=0.01   ，lstm的l2正则化惩罚系数

        训练参数：
        batch_size=64 训练时的batch_size，默认为64
        patience=50   训练时，在到达epochs前，当验证集上检测指标连续patience个epoch没有改进时，停止训练，默认为50
        epochs=1500     遍历训练集样本的次数，默认为1500




    '''

    path = r'\\TRADE302\DebugDir'
    #path = r'C:\wwj'

    model = rh3LearnGetModelDP003Unfit(time_window=time_window, features=features, num_class=num_class,
                                       depth=depth, use_residual=use_residual,
                                       bottleneck_size=bottleneck_size, kernel_size=kernel_size, nb_filters=nb_filters, pool_size=pool_size,
                                       lstm=lstm, lstm_units=lstm_units, l2=l2)

    log_name = os.path.join(path, "DP003.csv")
    print(log_name)

    hist = model.fit(train_x,
                     train_y,
                     batch_size=batch_size,
                     validation_data=(valid_x, valid_y),
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',
                                                              patience=patience, restore_best_weights=True),
                                keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, min_lr=0.0001)],
                     epochs=epochs)

    hist.history['epoch'] = hist.epoch
    train_proc = pd.DataFrame(hist.history)
    train_proc.to_csv(log_name)

    return model
