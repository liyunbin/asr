'''
Created on 2017年12月19日

The file is the implement of baidu deep speech 2.

'''

from keras.backend import ctc_batch_cost
from keras.layers import TimeDistributed, Conv2D, Input, Reshape, BatchNormalization, Dropout, \
GRU, LSTM, SimpleRNN, Dense, Lambda, Bidirectional, Activation
from keras.models import Model
import keras as K
from keras.optimizers import  sgd, adam
import asr_config
from keras.utils import plot_model


def ctc_lambda_function(args):
    y_true, y_pred, input_length, label_length = args;
    return ctc_batch_cost(y_true, y_pred, input_length, label_length);

def build_deepSpeech2(args):
    
    w = Input(shape=(args.max_time_steps, args.feature_num))
    
    # 增加通道维度，便于进行卷积操作
    x = Reshape(target_shape=(args.max_time_steps, args.feature_num, 1))(w)
    
    # 卷积层
    x = Conv2D(filters=32, kernel_size=(11, 41), strides=[2, 2], padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(args.keep_prob[0])(x)
    
    x = Conv2D(filters=32, kernel_size=(11, 21), strides=[2, 2], padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(args.keep_prob[1])(x)
    
    x = Conv2D(filters=96, kernel_size=(11, 21), strides=[2, 2], padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(args.keep_prob[2])(x)
    
    # 转换shape 方便rnn输入
    x = Reshape(target_shape=(x.shape[1].value, x.shape[2].value * x.shape[3].value))(x)
    
    # 7 recurrent layers
    # inputs must be [batch_size,timeSteps,feature]
    rnn = SimpleRNN;
    if args.rnn_type == 'GRU':
        rnn = GRU
    elif args.rnn_type == 'LSTM':
        rnn = LSTM  
    for i in range(args.rnn_stack_num):
        x = Bidirectional(rnn(args.num_hidden, return_sequences=True), merge_mode='sum')(x)
        x = Activation(activation='relu')(x) 
        x = BatchNormalization()(x)
        x = Dropout(args.keep_prob[3 + i])(x)

    # fully-connected layer
    x = TimeDistributed(Dense(args.num_hidden_fc, activation='softmax'))(x)
    
    base_model = Model(inputs=w, outputs=x);
    # 构建损失函数
    y_true = Input(shape=(args.max_char_len,))  # 样本label (batch,max_char_length)
    input_length = Input(shape=(1,))
    label_length = Input(shape=(1,));
    loss_out = Lambda(ctc_lambda_function, output_shape=(1,), name="ctc")([y_true, x, input_length, label_length])
    model = Model(inputs=[w, input_length, y_true, label_length], outputs=loss_out)
    optimizer = sgd(lr=0.0002, momentum=0.99, decay=.0, nesterov=True, clipnorm=400)
    model.compile(loss={'ctc': lambda y_true, y_pred: y_pred}, optimizer=optimizer)
    return base_model, model, optimizer, x.shape[1].value

  
if __name__ == '__main__':
    args = asr_config.deep_speech_2
    base_model, model, optimizer, final_timeSteps = build_deepSpeech2(args)
    print(model.summary())
    from keras.utils import plot_model
    plot_model(model, to_file='baidu_deep_speech2.png', show_shapes=True, show_layer_names=True)


    
