'''
Created on 2017年12月22日

@author: yunbin.li
'''
from keras.backend import ctc_batch_cost
from keras.models import load_model 

import asr_config
import data_utils
import deep_speech_2
import keras as K
import numpy as np
import tensorflow as tf
import residual_conv_ctc
import data_utils
import time
import  os
import  pandas as pd



def predict(model, index_char, wave_path, args):
    orgin_data = None
    if args.feature_type == 'spect':
        orgin_data = data_utils.extract_spectrogram(wave_path, args)
    else:
        orgin_data = data_utils.extract_mfcc(wave_path, args.sample_rate, args.feature_num)
    data = data_utils.expand_timeSteps(orgin_data, args.max_time_steps)
    input_vec = np.reshape(data, newshape=(1, data.shape[0], data.shape[1]))
    t1 = time.time()
    y_pred = model.predict(input_vec)
    t2 = time.time()
    print('预测时间：{}'.format(t2 - t1))
    # print(y_pred.shape)
    ctc_decodes = K.backend.ctc_decode(y_pred, greedy=True, beam_width=100, top_paths=10,
                                       input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1])[0]
    print(len(ctc_decodes))
    for path in ctc_decodes:
        out = K.backend.get_value(path);
        # print(out.shape)
        answer = [index_char[i] for i in out[0] if i < len(index_char)];
        final_chinese = "".join(answer);
        # print(final_chinese)
    return final_chinese;

if __name__ == '__main__':
    args = asr_config.tencent_speech
    char_index_dict, index_char_dict = asr_config.char_index_dict, asr_config.index_char_dict
    # sbase_model, model = deep_speech_2.build_deepSpeech2(args)
    base_model, model, out_timeSteps = residual_conv_ctc.build_model(args)
    base_model.load_weights('checkout/tencent_speech_epoch.weights.010-83.35.hdf5')
    data_dir = '/home/lyb/asr/data/data_thchs30/test'
    
    graph = tf.get_default_graph()
    with graph.as_default():
        file_names = os.listdir(data_dir)
        
        with open('result.txt', mode='w', encoding='utf-8') as rw:
            for name in file_names:
                if name.endswith('.wav'):
                    wave_path = os.path.join(data_dir, name)
                    trn_content = list(open(os.path.join(data_dir, name + '.trn'), mode='r').readlines())[0].strip('\n')
                    # print(trn_content)
                    trn_path = os.path.join(data_dir, trn_content)
                    # print(trn_path)
                    corpus = list(open(trn_path, mode='r').readlines())[0]
                    corpus = ' '.join(''.join(corpus.strip('\n').split(' ')))  # 每个汉字以空格相连
                    rw.write('raw:{}\n'.format(corpus))
                    print('raw:{}'.format(corpus))
                    t1 = time.time()
                    
                    pred = predict(base_model, index_char_dict, wave_path, args)
                    rw.write('pred:{}\n'.format(pred))
                    print('pred:{}'.format(pred))
                    
                    t2 = time.time()
                    print('预测总耗时：{}'.format(t2 - t1))
                    
                    rw.flush()




    

