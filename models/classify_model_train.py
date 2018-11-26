'''
Created on 2017年11月24日

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
from sklearn.model_selection import KFold, StratifiedKFold
import os
from keras.utils import to_categorical

class Performance(Callback):
    
    def __init__(self, inputs, y_true, model, kfold='all', algo=None, log_dir='.'):
        """
        :Params
            inputs: speech features, such as "mfcc". shape is [samples, timesteps, feature_num]
            y_true: transcript of the audio. chars encoder by ids. shape is [samples, max_tokens]
            
        """
        self.inputs = inputs
        self.y_true = y_true
        self.model = model
        self.kfold= kfold
        self.algo=algo
        self.log_dir = log_dir
        path = log_dir+'/result'
        if not os.path.exists(path):
            os.mkdir(path)
        
    def on_epoch_end(self, epoch, logs=None):
        print('The current epoch number: {}'.format(epoch))
        if self.inputs is not None:
            total_samples = self.inputs.shape[0]
            probs = self.model.predict(self.inputs, batch_size=200)
            if isinstance(probs, list):
                pred_numbers = np.hstack([np.argsort(n, axis=1)[::, -1].reshape((n.shape[0],1)) for n in probs])
            else:
                pred_numbers = np.argmax(probs, axis=1)
            counter = 0
            for i in range(self.y_true.shape[0]):
                if self.y_true[i].tolist() == pred_numbers[i].tolist():
                    counter += 1
            with open(self.log_dir+"/result/test_{}_{}.log".format(self.algo, self.kfold), 'a') as f:
                f.write('{}\t{}\n'.format(epoch, counter/float(total_samples)))
            print('the total samples:{}, accuracy are:{}'.format(total_samples, counter/float(total_samples)))
            
                
def train_4(train_corpus_dir, args):
    """分类模型，输出4为验证码
    """

    #加载数据
    (inputs, input_length, y_true, label_length) = \
    data_utils.load_data(train_corpus_dir, args.max_time_steps)
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
        dev_performance_labels = y_true[validate_idx]
        
        model = classify_model.build_vanilla(args)
        print(model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        model_checkpoint = ModelCheckpoint(filepath=args.check_point +'/'+ 'classify_model-fold_{}.hdf5'.format(kf_num),
                                           monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=False)
        performance = Performance(inputs=dev_inputs, y_true=dev_performance_labels, 
                        model=model, kfold=kf_num, 
                        algo='deepspeech2_for_4', log_dir=args.check_point)
        history = model.fit(
                x=train_inputs,
                y=[train_num_1, train_num_2, train_num_3, train_num_4],
                validation_data=(dev_inputs, [dev_num_1, dev_num_2, dev_num_3, dev_num_4]),
                epochs=args.nb_epoch,
                batch_size=args.batch_size, shuffle=False,
                callbacks=[model_checkpoint, performance, early_stopping])
        kf_num += 1
        
def train_single(train_corpus_dir, args):
    """分类模型，输出为单个数字
    """
    #加载数据
    (inputs, input_length, y_true, label_length) = \
    data_utils.load_data(train_corpus_dir, args.max_time_steps)
    y_labels = to_categorical(y_true, args.num_hidden_fc)
    
    kf = StratifiedKFold(n_splits=5)
    kf_num = 1
    if not os.path.exists(args.check_point):
        os.makedirs(args.check_point)
    for train_idx, validate_idx in kf.split(inputs, y_true):
        # 测试集
        train_inputs = inputs[train_idx]
        train_labels = y_labels[train_idx] 
  
        #验证集
        dev_inputs =  inputs[validate_idx]
        dev_labels = y_labels[validate_idx]
        dev_performance_labels = y_true[validate_idx].flatten()
                
        model = classify_model.build_single_vanilla(args)
        print(model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=100)
        model_checkpoint = ModelCheckpoint(filepath=args.check_point +'/'+ 'classify_model-fold_{}.hdf5'.format(kf_num),
                                           monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=False)
        performance = Performance(inputs=dev_inputs, y_true=dev_performance_labels, 
                                  model=model, kfold=kf_num, 
                                  algo='deepspeech2_for_single', log_dir=args.check_point)
        
        history = model.fit(
                x=train_inputs,
                y=train_labels,
                validation_data=(dev_inputs, dev_labels),
                epochs=args.nb_epoch,
                batch_size=args.batch_size, shuffle=False,
                callbacks=[performance, model_checkpoint, early_stopping])
        kf_num += 1
    

if __name__ == '__main__':
    gpu_num = "1"
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_num
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))
    classify_config = asr_config.classify_config
    train_4('../classify_features',classify_config)
#     classify_single_config = asr_config.classify_single_config
#     train_single('../single_features',classify_single_config)





