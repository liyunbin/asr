'''
Created on 2017年12月26日


The implementation of residual Convolutional CTC Networls.
The paper is : Residual Convolutional CTC Networks for Automatic Speech Recognition

'''


from keras.layers import TimeDistributed, Conv2D, Input, Reshape, BatchNormalization, Dropout, \
GRU, LSTM, SimpleRNN, Dense, Lambda, Bidirectional, Activation, Add
from keras.models import Model
from keras.optimizers import  sgd, adam
from keras.regularizers import l2
from deep_speech_2 import  ctc_lambda_function
import keras as K
import tensorflow as tf
from asr_config import tencent_speech
from keras.utils import plot_model
from dask.array.ufunc import sign


# Wide residual network http://arxiv.org/abs/1605.07146
def _wide_basic(n_input_plane, n_output_plane, stride):
    def f(net):
        # format of conv_params:
        #               [ [nb_col="kernel width", nb_row="kernel height",
        #               subsample="(stride_vertical,stride_horizontal)",
        #               border_mode="same" or "valid"] ]
        # B(3,3): orignal <<basic>> block
        conv_params = [ [(3, 3), stride, "same"],
                        [(3, 3), (1, 1), "same"]]
        
        n_bottleneck_plane = n_output_plane

        # Residual block
        for i, v in enumerate(conv_params):
            if i == 0:
                if n_input_plane != n_output_plane:
                    net = BatchNormalization(axis=-1)(net)
                    net = Activation("relu")(net)
                    convs = net
                else:
                    convs = BatchNormalization(axis=-1)(net)
                    convs = Activation("relu")(convs)
                convs = Conv2D(n_bottleneck_plane, kernel_size=v[0],
                                     strides=v[1],
                                     padding=v[2],
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=l2(.0005)
                                     )(convs)
            else:
                convs = BatchNormalization(axis=-1)(convs)
                convs = Activation("relu")(convs)
                # convs = Dropout(0.5)(convs)
                convs = Conv2D(n_bottleneck_plane, kernel_size=v[0],
                                     strides=v[1],
                                     padding=v[2],
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=l2(.0005)
                                     )(convs)
        # Shortcut Conntection: identity function or 1x1 convolutional
        #  (depends on difference between input & output shape - this
        #   corresponds to whether we are using the first block in each
        #   group; see _layer() ).
        if n_input_plane != n_output_plane:
            shortcut = Conv2D(n_output_plane, kernel_size=(1, 1),
                                     strides=stride,
                                     padding='same',
                                     kernel_initializer='he_normal',
                                     kernel_regularizer=l2(.0005)
                                     )(net)
        else:
            shortcut = net
            
        return Add()([convs, shortcut])
    
    return f


# "Stacking Residual Units on the same stage"
def _layer(block, n_input_plane, n_output_plane, count, stride):
    def f(net):
        net = block(n_input_plane, n_output_plane, stride)(net)
        for i in range(2, int(count + 1)):
            net = block(n_output_plane, n_output_plane, stride=(1, 1))(net)
        return net
    
    return f  

def build_model(args):
    
    # 输入 [batch,timeSteps,featureNum]
    x = Input(shape=(args.max_time_steps, args.feature_num))
   
    # 增加通道维度，便于进行卷积操作
    w = Reshape(target_shape=(args.max_time_steps, args.feature_num, 1))(x)
    
    k = 2  # widen_factor
    n_stages = [32, 64 * k, 128 * k, 256 * k, 512 * k ]
    strides = [(1, 1), (1, 1), (1, 2), (2, 2)]
    count = [2, 2, 2, 2]  # 每一个resBlock的数量
    
    conv = Conv2D(filters=n_stages[0],
                   kernel_size=(11, 41),
                   strides=(2, 2),
                   padding="same",
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(.0005),
                   )(w)
    block_fn = _wide_basic
    
    for i in range(len(n_stages) - 1):
        print('当前stage:{}'.format(i))
        conv = _layer(block_fn, n_input_plane=n_stages[i], n_output_plane=n_stages[i + 1], count=count[i], stride=strides[i])(conv)  # "Stage 1 (spatial size: 32x32)"
        
    batch_norm = BatchNormalization(axis=-1)(conv)
    
    # 转换shape channel
    out = Reshape(target_shape=(batch_norm.shape[1].value, batch_norm.shape[2].value * batch_norm.shape[3].value))(batch_norm)
    
    # fully-connected layer
    out = TimeDistributed(Dense(args.num_hidden_fc, activation='softmax'))(out)
   
    base_model = Model(inputs=x, outputs=out);
    
    # 构建损失函数
    y_true = Input(shape=(args.max_char_len,))  # 样本label (batch,max_char_length)
    input_length = Input(shape=(1,))
    label_length = Input(shape=(1,));
    loss_out = Lambda(ctc_lambda_function, output_shape=(1,), name="ctc")([y_true, out, input_length, label_length])
    model = Model(inputs=[x, input_length, y_true, label_length], outputs=loss_out)
    optimizer = sgd(lr=0.0001)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    return base_model, model, out.shape[1].value
    
if __name__ == '__main__':
    base_model, model, final_timeSteps = build_model(tencent_speech)
    print(model.summary())
    print(final_timeSteps)
    plot_model(model, to_file='model.png', show_shapes=True, show_layer_names=True)
    from scipy import signal    

    
