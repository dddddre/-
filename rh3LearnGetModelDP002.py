# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function, unicode_literals

from tensorflow import keras
import tensorflow as tf

import os
import pandas as pd
__all__=['rh3LearnGetModelDP002', 'rh3LearnGetModelDP002Unfit']


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


def inception_net(input_tensor,
                  depth=6, use_residual=True,
                  bottleneck_size=32, kernel_size=40, nb_filters=32, pool_size=3,
                  ):
    '''
    function:该模型为InceptionTime模型，
    https://github.com/hfawaz/InceptionTime/blob/master/classifiers/inception.py
    author:@wwj
    input:
        该模型特定参数：
        depth=6  网络总深度，一般取3的正整数倍
        use_residual=True 是否使用使用resNet
        bottleneck_size=32 inceptionmodule中的bottleneck的filters数
        kernel_size=40 inceptionmodule中的三个窗口的上限
        nb_filters=32  inceptionmodule中的三个窗口的filters数
        pool_size=3    inceptionmodule中的maxpool的size

    '''

    x = input_tensor
    input_res = input_tensor

    for d in range(depth):
        x = inception_module(x, bottleneck_size=bottleneck_size,
                             kernel_size=kernel_size, nb_filters=nb_filters, pool_size=pool_size)
        if use_residual and d % 3 == 2:
            x = shortcut_layer(input_res, x)
            input_res = x
    return x


class InputTranLayer(keras.layers.Layer):
        '''用来对输入数据按照要求排序'''

        def __init__(self):
            super(InputTranLayer, self).__init__()

        def build(self, input_shape):
            self.time_window = input_shape[1]
            self.var_num = input_shape[2]

            self.a = tf.scatter_nd(
                tf.constant(
                    [[11, 0], [1, 1], [19, 2], [9, 3], [18, 4], [8, 5], [17, 6], [7, 7], [16, 8], [6, 9], [15, 10],
                     [5, 11],
                     [14, 12], [4, 13], [13, 14], [3, 15], [12, 16], [2, 17], [10, 18], [0, 19], [20, 20], [30, 21],
                     [22, 22], [32, 23], [23, 24], [33, 25], [24, 26], [34, 27], [25, 28], [35, 29], [26, 30], [36, 31],
                     [27, 32], [37, 33], [28, 34], [38, 35], [29, 36], [39, 37], [21, 38], [31, 39], ], dtype=tf.int32),
                tf.ones((self.var_num,)),
                tf.constant([self.var_num, self.var_num]))

        def call(self, inputs):
            temp = tf.matmul(tf.reshape(inputs, [-1, self.var_num]), self.a)
            return tf.reshape(temp, [-1, self.time_window, self.var_num])


class SliceLayer(keras.layers.Layer):
        '''该类用来截取特定档位的数据'''
        def __init__(self,start, stop, if_reverse):
            super(SliceLayer, self).__init__()
            self.start=start
            self.stop=stop
            self.if_reverse=if_reverse

        def call(self, inputs):
            if self.if_reverse:
                return tf.reverse(inputs[:, :, self.start:self.stop], axis=[2])
            else:
                return inputs[:, :, self.start:self.stop]


class SymConv2D(keras.layers.Layer):
        '''该类用来做对称卷积'''
        def __init__(self,units,reg=None):
            super(SymConv2D, self).__init__()
            self.units = units
            self.reg = reg

        def build(self, input_shape):
            self.level = input_shape[-2]
            self.halflevel = int(self.level / 2)
            if self.level % 2 == 0:
                self.temp_layer = keras.layers.Conv2D(self.units, (1, self.halflevel), kernel_regularizer=self.reg)
                self.slice_layer_up = SliceLayer(0, self.halflevel, False)
                self.slice_layer_down = SliceLayer(self.halflevel,self.level,True)
            else:
                self.temp_layer = keras.layers.Conv2D(self.units, (1, self.halflevel+1), kernel_regularizer=self.reg)
                self.slice_layer_up = SliceLayer(0, self.halflevel+1, False)
                self.slice_layer_down = SliceLayer(self.halflevel,self.level,True)

        def call(self, inputs):
            xup = self.slice_layer_up(inputs)
            xdown = self.slice_layer_down(inputs)
            xup = self.temp_layer(xup)
            xdown = self.temp_layer(xdown)
            return xup+xdown


def rh3LearnGetModelDP002Unfit(time_window=200, features=40, num_class=3,
                               t_window1=2, level_window1=3, level_window2=3,
                               units1=64, units2=32, units3=32, units4=32, units5=64, units6=128,
                               use_inception=False, BN=False,
                               short=1, mid=5, long=10,
                               l2=0.0, inputdroprate=0.2, hiddendroprate=0.5, **kwargs):
    '''
    function:该模型为四层卷积加上LSTM的神经网络，没有训练的模型
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
        t_window1 = 2  第一层2维卷积核的时间维度大小
        level_window1 = 3  # 第一层2维卷积核的档位维度大小，为大于0的奇数
        level_window2 = 3  # 中间部分的档位窗口大小，大于0小于等于10
        units1 = 64  第一层卷积的filters
        units2 = 32  第二层卷积的filters
        units3 = 32  第三层卷积的filters
        units4 = 32  第四城卷积(时间上卷积)或inception的filters
        units5 = 64  lstm的输出单元数
        units6 = 128  全连接的输出单元数
        use_inception=False 是否使用inception
        BN=False 时间卷积前是否使用batch normalization
        short = 1    第四次卷积的短窗口
        mid = 3      第四次卷积的中窗口
        long = 5     第四次卷积的长窗口
        l2=0.0      l2正则化
        inputdroprate=0.2 输入层的dropout rate
        hiddendroprate=0.5  隐层的dropout rate

    '''

    units2 = int(units2)
    units3 = int(units3)

    level_num = int(features / 2)

    inputs = keras.Input(shape=[time_window, features])  # 生成输入的占位符
    x = InputTranLayer()(inputs)  # 对40个features按规定顺序排序
    x = keras.layers.Dropout(inputdroprate)(x)
    x = keras.layers.Reshape((time_window, level_num, 2))(x)  # 将样本的shape变为(time_window,level_num,channel),
    # level_num是从最高价到最低价排下来，channel为2，表示量和价
    x = keras.layers.ZeroPadding2D(((t_window1 - 1, 0), (int((level_window1 - 1) / 2), int((level_window1 - 1) / 2))))(
        x)  # 在时间维度上，开头补0，因为接下来时间维度上的卷积会消耗掉一部分
    x = keras.layers.Conv2D(units1, (t_window1, level_window1), activation='relu',
                            kernel_regularizer=keras.regularizers.l2(l2)
                            )(x)          # 做二维卷积，卷积核大小为(t_window1, level_window1)
    x = keras.layers.Dropout(hiddendroprate)(x)
    x1 = SliceLayer(0, int(level_num / 2), False)(x)  # 取出档位中的卖的部分
    x2 = SliceLayer(int(level_num / 2), level_num, True)(x)  # 取出档位中的买的部分
    x3 = SliceLayer(int(level_num / 2) - level_window2, int(level_num / 2) + level_window2,False)(x)
    # 取出档位中的中间部分，即level_window2档买和level_window2档卖
    convLayer = keras.layers.Conv2D(units2, (1, int(level_num / 2)), activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(l2))  # 生成对买或卖的卷积，买和卖用相同的卷积核
    x1 = convLayer(x1)
    x2 = convLayer(x2)
    x3 = SymConv2D(units2, reg=keras.regularizers.l2(l2))(x3)  # 对中间部分做卷积，约束买卖对称
    x3 = tf.keras.layers.Activation('relu')(x3)
    x = keras.layers.concatenate([x1, x3, x2], axis=-2)  # 将卖、中间、买三块并起来

    x = keras.layers.Dropout(hiddendroprate)(x)
    x = SymConv2D(units3, reg=keras.regularizers.l2(l2))(x)  # 做卷积，约束买卖对称
    x = tf.keras.layers.Activation('relu')(x)
    x = keras.layers.Reshape((time_window, units3))(x)  # 将档位维度去除

    x = keras.layers.Dropout(hiddendroprate)(x)
    if not use_inception:
        if BN:
            x = keras.layers.BatchNormalization()(x)

        xshort = keras.layers.Conv1D(units4, short, activation='relu',
                                     kernel_regularizer=keras.regularizers.l2(l2))(
                                keras.layers.ZeroPadding1D((short - 1, 0))(x))  # 时间上卷积，用较短的卷积核
        xmid = keras.layers.Conv1D(units4, mid, activation='relu',
                                   kernel_regularizer=keras.regularizers.l2(l2))(
                                   keras.layers.ZeroPadding1D((mid - 1, 0))(x))  # 时间上卷积，用中等长度的卷积核
        xlong = keras.layers.Conv1D(units4, long, activation='relu',
                                    kernel_regularizer=keras.regularizers.l2(l2))(
                                    keras.layers.ZeroPadding1D((long - 1, 0))(x))  # 时间上卷积，用较长的卷积核
        x = keras.layers.concatenate([xshort, xmid, xlong], axis=-1)  # 将不同长度的并起来
    else:
        x = inception_net(x, nb_filters=units4)

    x = keras.layers.LSTM(units5, dropout=hiddendroprate, recurrent_dropout=hiddendroprate,
                          kernel_regularizer=keras.regularizers.l2(l2),
                          recurrent_regularizer=keras.regularizers.l2(l2)
                          )(x)
    #x = keras.layers.RNN(keras.layers.LSTMCell(units5,
    #                                           dropout=hiddendroprate,
    #                                           recurrent_dropout=hiddendroprate))(x)
     # 对时间序列做LSTM

    x = keras.layers.Dropout(hiddendroprate)(x)

    x = keras.layers.Dense(units6, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(l2))(x)  # 加全连接层

    if l2 > 1e-6:
        x = keras.layers.Dropout(0.5)(x)

    output = keras.layers.Dense(num_class, activation='softmax')(x)  # 分类

    model = keras.Model(inputs=inputs, outputs=output, name='model1')
    #model.summary()
    #keras.utils.plot_model(model, "nclass_{}_L2_{}_units2_{}_inception_{}_BN_{}_inputdroprate_{}_hiddendroprate_{}.png".
     #                      format(num_class, l2, units2, use_inception, BN, inputdroprate, hiddendroprate), show_shapes=True)

    model.compile(loss=keras.losses.CategoricalCrossentropy(),
                  optimizer=keras.optimizers.Adam(),
                  metrics=[keras.metrics.CategoricalAccuracy()])
    return model


def rh3LearnGetModelDP002(train_x, train_y, valid_x, valid_y,
                          time_window=200,
                          features=40,
                          num_class=3,
                          t_window1=2,
                          level_window1=3,
                          level_window2=3,
                          units1=64,
                          units2=32,
                          units3=32,
                          units4=32,
                          units5=64,
                          units6=128,
                          use_inception=False,
                          BN=False,
                          short=1,
                          mid=5,
                          long=10,
                          l2=0.0,
                          inputdroprate=0.2,
                          hiddendroprate=0.5,
                          batch_size=64,
                          patience=50,
                          epochs=1000,
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
        t_window1 = 2  第一层2维卷积核的时间维度大小
        level_window1 = 3  # 第一层2维卷积核的档位维度大小，为大于0的奇数
        level_window2 = 3  # 中间部分的档位窗口大小，大于0小于等于10
        units1 = 64  第一层卷积的filters
        units2 = 32  第二层卷积的filters
        units3 = 32  第三层卷积的filters
        units4 = 32  第四层卷积(时间上卷积)或inception的filters
        units5 = 64  lstm的输出单元数
        units6 = 64  全连接的输出单元数
        use_inception=False 是否使用inception
        BN=False 是否使用BatchNormalisation
        short = 1    第四次卷积的短窗口
        mid = 3      第四次卷积的中窗口
        long = 5     第四次卷积的长窗口
        正则化参数：
        l2=0.0      l2正则化
        inputdroprate=0.2 输入层的dropout rate
        hiddendroprate=0.5  隐层的dropout rate
        训练参数：
        batch_size 训练时的batch_size，默认为64
        patience   训练时，在到达epochs前，当验证集上检测指标连续patience个epoch没有改进时，停止训练，默认为50
        epochs     遍历训练集样本的次数，默认为1000




    '''

    path = r'\\TRADE302\DebugDir'
    #path = r'C:\wwj'

    units2 = int(units2)
    units3 = int(units3)
    model = rh3LearnGetModelDP002Unfit(time_window=time_window, features=features, num_class=num_class,
                                       t_window1=t_window1, level_window1=level_window1, level_window2=level_window2,
                                       units1=units1, units2=units2, units3=units3, units4=units4, units5=units5, units6=units6,
                                       use_inception=use_inception, BN=BN,
                                       short=short, mid=mid, long=long,
                                       l2=l2, inputdroprate=inputdroprate, hiddendroprate=hiddendroprate)

    log_name = os.path.join(path, "nclass_{}_L2_{}_units2_{}_inception_{}_BN_{}_inputdroprate_{}_hiddendroprate_{}.csv".
                            format(num_class, l2, units2, use_inception, BN, inputdroprate, hiddendroprate))
    print(log_name)
    hist = model.fit(train_x,
                     train_y,
                     batch_size=batch_size,
                     validation_data=(valid_x, valid_y),
                     callbacks=[keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy', patience=patience,
                                                              restore_best_weights=True),
                                ],
                     epochs=epochs)

    hist.history['epoch'] = hist.epoch
    train_proc = pd.DataFrame(hist.history)
    train_proc.to_csv(log_name)

    return model
