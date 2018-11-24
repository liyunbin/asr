'''
Created on 2017年12月21日

@author: yunbin.li

'''
from keras.backend.tensorflow_backend import set_session
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
from keras.models import save_model
import asr_config
import data_utils
import deep_speech_2
import numpy as np
import tensorflow as tf
import keras as  K
from sklearn.model_selection import KFold
import os

class Performance(Callback):
    
    def __init__(self, inputs, y_true, base_model):
        self.inputs = inputs
        self.y_true = y_true
        self.base_model = base_model
        
    def on_epoch_end(self, epoch, logs=None):
        pass

class SaveModel(Callback):
    
    def __init__(self, filepath, base_model, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1):
        super(SaveModel, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.base_model = base_model

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.base_model.save_weights(filepath, overwrite=True)
                        else:
                            self.base_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve' %
                                  (epoch + 1, self.monitor))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.base_model.save_weights(filepath, overwrite=True)
                else:
                    self.base_model.save(filepath, overwrite=True)
                    

def train(train_corpus_dir, args):

    #加载数据
    base_model, model, optimizer, final_timeSteps = deep_speech_2.build_deepSpeech2(args)
    (inputs, input_length, y_true, label_length) = \
    data_utils.load_data(train_corpus_dir, final_timeSteps)
    kf = KFold(n_splits=5)
    kf_num = 1
    if not os.path.exists(args.check_point):
        os.makedirs(args.check_point)
    for train_idx, validate_idx in kf.split(y_true):
        # 测试集
        train_inputs = inputs[train_idx]
        train_input_length = input_length[train_idx]
        train_y_true = y_true[train_idx]
        train_label_length = label_length[train_idx]
        
        #验证集
        dev_inputs =  inputs[validate_idx]
        dev_input_length = input_length[validate_idx]
        dev_y_true = y_true[validate_idx]
        dev_label_length = label_length[validate_idx]
        
        base_model, model, optimizer, final_timeSteps = deep_speech_2.build_deepSpeech2(args)
        print(model.summary())
        early_stopping = EarlyStopping(monitor='val_loss', patience=20)
        model_checkpoint = SaveModel(filepath=args.check_point +'/'+ 'fold-{}'.format(kf_num) +'deep_speech2_epoch.{epoch:03d}-{val_loss:.2f}.hdf5',
                                           base_model=base_model,
                                           monitor='val_loss',
                                           save_best_only=True, 
                                           save_weights_only=False)
        history = model.fit(
                x=[train_inputs, train_input_length, train_y_true, train_label_length],
                y=np.ones(train_y_true.shape[0]),
                validation_data=([dev_inputs, dev_input_length, dev_y_true, dev_label_length], np.ones(dev_y_true.shape[0])),
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
    deep_speech2_config = asr_config.deep_speech_2
    train('../features',deep_speech2_config)





