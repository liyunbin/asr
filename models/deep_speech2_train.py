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

class SaveBaseModel(Callback):
    
    def __init__(self, filepath, base_model, optimizer):
        self.filepath = filepath
        self.base_model = base_model
        self.optimizer = optimizer
        
    def on_epoch_end(self, epoch, logs=None):
        K.backend.set_value(self.optimizer.lr, 1.2 * K.backend.get_value(self.optimizer.lr))
        if epoch % 5 == 0:
            file_path = self.filepath.format(epoch=epoch, **logs)
            print('保存baseModel,epoch:{},path:{}'.format(epoch, file_path))
            self.base_model.save_weights(file_path)

# train_corpus_dir = '/home/lyb/asr/data/data_thchs30/train'
# dev_corpus_dir = '/home/lyb/asr/data/data_thchs30/dev'
# test_corpus_dir = '/home/lyb/asr/data/data_thchs30/test'
train_corpus = 'spect_train'
dev_corpus = 'spect_dev'
test_corpus = 'spect_test'


# 模型参数设置
args = asr_config.deep_speech_2
char_index_dict, index_char_dict = asr_config.char_index_dict, asr_config.index_char_dict

# 加载模型
base_model, model, optimizer, final_timeSteps = deep_speech_2.build_deepSpeech2(args)
print(model.summary())

# 加载数据集
(train_inputs, train_input_length, train_y_true, train_label_length) = data_utils.load_data(train_corpus, final_timeSteps)
(dev_inputs, dev_input_length, dev_y_true, dev_label_length) = data_utils.load_data(dev_corpus, final_timeSteps)
# (test_inputs, test_input_length, test_y_true, test_label_length) = data_utils.thchs_data_process(test_corpus_dir,
#                                                                               char_index_dict,
#                                                                               args.max_char_len, args.max_time_steps,
#                                                                               args.sample_rate,
#                                                                               args.feature_num)

def train(args):

    # early_stopping = EarlyStopping(monitor='val_loss', patience=100)
    save_base_model = SaveBaseModel(args.check_point + '/' + 'deep_speech2_epoch.weights.{epoch:03d}-{val_loss:.2f}.hdf5',
                                     base_model,
                                     optimizer)
#     model_checkpoint = ModelCheckpoint(args.check_point + '/' + 'deep_speech2_epoch.{epoch:03d}-{val_loss:.2f}.hdf5',
#                                        save_best_only=True, save_weights_only=True)
    history = model.fit(
            x=[train_inputs, train_input_length, train_y_true, train_label_length],
            y=np.ones(train_y_true.shape[0]),
            validation_data=([dev_inputs, dev_input_length, dev_y_true, dev_label_length], np.ones(dev_y_true.shape[0])),
            epochs=args.nb_epoch,
            batch_size=args.batch_size, shuffle=False,
            callbacks=[save_base_model])
    save_model(base_model, './baidu_deep_speech_2.model')
    

if __name__ == '__main__':
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.8
    set_session(tf.Session(config=config))
    train(args)





