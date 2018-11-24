'''
Created on 2017年11月24日

@author: yunbin.li

'''
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import save_model
import asr_config
import data_utils
import classify_model
import numpy as np
import tensorflow as tf
import keras as  K
from sklearn.model_selection import KFold
import os
from keras.utils import to_categorical

class Performance(Callback):
    
    def __init__(self, inputs, y_true, base_model):
        self.inputs = inputs
        self.y_true = y_true
        self.base_model = base_model
        
    def on_epoch_end(self, epoch, logs=None):
        pass



def train(train_corpus_dir, args):

    #加载数据
    (inputs, input_length, y_true, label_length) = \
    data_utils.load_data(train_corpus_dir, args.max_time_steps)
    y_true = y_true - 1 # 分类从0开始编码
    num_1 = y_true[::, 0]
    num_2 = y_true[::, 1]
    num_3 = y_true[::, 2]
    num_4 = y_true[::, 3]
    num_1 = to_categorical(num_1, args.num_hidden_fc)
    num_2 = to_categorical(num_2, args.num_hidden_fc)
    num_3 = to_categorical(num_3, args.num_hidden_fc)
    num_4 = to_categorical(num_4, args.num_hidden_fc)
    
    kf = KFold(n_splits=5)
    kf_num = 1
    if not os.path.exists(args.check_point):
        os.makedirs(args.check_point)
    for train_idx, validate_idx in kf.split(y_true):
        # 测试集
        train_inputs = inputs[train_idx]
        train_num_1 = num_1[train_idx] 
        train_num_2 = num_2[train_idx] 
        train_num_3 = num_3[train_idx] 
        train_num_4 = num_4[train_idx] 
  
        #验证集
        dev_inputs =  inputs[validate_idx]
        dev_num_1 = num_1[validate_idx] 
        dev_num_2 = num_2[validate_idx] 
        dev_num_3 = num_3[validate_idx] 
        dev_num_4 = num_4[validate_idx] 
        
        model = classify_model.build_vanilla(args)
        print(model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        model_checkpoint = ModelCheckpoint(filepath=args.check_point +'/'+ 'classify_model-fold_{}.hdf5'.format(kf_num),
                                           monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=False)
        history = model.fit(
                x=train_inputs,
                y=[train_num_1, train_num_2, train_num_3, train_num_4],
                validation_data=(dev_inputs, [dev_num_1, dev_num_2, dev_num_3, dev_num_4]),
                epochs=args.nb_epoch,
                batch_size=args.batch_size, shuffle=False,
                callbacks=[model_checkpoint, early_stopping])
        kf_num += 1
    

if __name__ == '__main__':
    gpu_num = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    classify_config = asr_config.classify_config
    train('../features',classify_config)





