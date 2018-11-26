'''
Created on 2018年11月24日

验证码分类模型实现

@author: yunbin.li
'''

from keras.layers import TimeDistributed, Conv2D, Input, Reshape, BatchNormalization, Dropout, \
GRU, LSTM, SimpleRNN, Dense, Lambda, Bidirectional, Activation
from keras.models import Model
import keras as K
from keras.optimizers import  sgd, adam
import asr_config



def build_vanilla(args):
    
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
        if i != args.rnn_stack_num-1:
            x = Bidirectional(rnn(args.num_hidden, return_sequences=True), merge_mode='sum')(x)
            x = Activation(activation='relu')(x) 
            x = BatchNormalization()(x)
            x = Dropout(args.keep_prob[3 + i])(x)
        else:
            x = Bidirectional(rnn(args.num_hidden, return_sequences=False), merge_mode='sum')(x)
            x = Activation(activation='relu')(x) 
            x = BatchNormalization()(x)
            x = Dropout(args.keep_prob[3 + i])(x) 

    num_1 = Dense(units=args.num_hidden_fc, activation='softmax',name='num_1')(x)
    num_2 = Dense(units=args.num_hidden_fc, activation='softmax', name='num_2')(x)
    num_3 = Dense(units=args.num_hidden_fc, activation='softmax', name='num_3')(x)
    num_4 = Dense(units=args.num_hidden_fc, activation='softmax', name='num_4')(x)
        
    optimizer = sgd(lr=0.01, momentum=0.99, decay=.0, nesterov=True, clipnorm=400)
    model = Model(inputs=w, outputs=[num_1, num_2, num_3, num_4]);
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

def build_single_vanilla(args):
    
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
        if i != args.rnn_stack_num-1:
            x = Bidirectional(rnn(args.num_hidden, return_sequences=True), merge_mode='sum')(x)
            x = Activation(activation='relu')(x) 
            x = BatchNormalization()(x)
            x = Dropout(args.keep_prob[3 + i])(x)
        else:
            x = Bidirectional(rnn(args.num_hidden, return_sequences=False), merge_mode='sum')(x)
            x = Activation(activation='relu')(x) 
            x = BatchNormalization()(x)
            x = Dropout(args.keep_prob[3 + i])(x) 

    num = Dense(units=args.num_hidden_fc, activation='softmax',name='num')(x)        
    optimizer = sgd(lr=0.0002, momentum=0.99, decay=.0, nesterov=True, clipnorm=400)
    model = Model(inputs=w, outputs=num);
    model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

  
if __name__ == '__main__':
    args = asr_config.deep_speech_2
    model = build_vanilla(args)
    print(model.summary())


    
