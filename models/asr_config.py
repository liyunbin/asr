import data_utils
import numpy as np

class Struct:
    
    def __init__(self, **entries): 
        self.__dict__.update(entries)

       
# 模型参数这设置
baidu_deep_speech_2 = {
    'window_size':.025,  # 帧长短 单位：秒
    'window_stride': .01,  # 帧窗口 单位：秒
    'window':np.hamming,
    'feature_normalize':True,
    'feature_type':'mfcc',
    'feature_num':13,
    'sample_rate':44100,
    'max_time_steps':2000,
    'keep_prob':[.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
    'num_hidden':800,
    'num_hidden_fc':10,  # 及就是多少个中文字符，简体字加上罗马字母表加上空格
    'max_char_len':4,
    'rnn_type':'GRU',
    'rnn_stack_num':3,
    'nb_epoch':1000,
    'batch_size':40,
    'check_point':'./checkout'
    }


classify_dict = {
    'window_size':.025,  # 帧长短 单位：秒
    'window_stride': .01,  # 帧窗口 单位：秒
    'window':np.hamming,
    'feature_normalize':True,
    'feature_type':'mfcc',
    'feature_num':13,
    'sample_rate':44100,
    'max_time_steps':2000,
    'keep_prob':[.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
    'num_hidden':800,
    'num_hidden_fc':10,  # 及就是多少个中文字符，简体字加上罗马字母表加上空格
    'max_char_len':4,
    'rnn_type':'GRU',
    'rnn_stack_num':3,
    'nb_epoch':1000,
    'batch_size':40,
    'check_point':'./classify_checkpoints'
    }

tencent_speech_dict = {
    'window_size':.02,  # 帧长短 单位：秒
    'window_stride': .01,  # 帧窗口 单位：秒
    'window':'hamming',
    'feature_normalize':True,
    'feature_type':'spect',
    'feature_num':161,
    'sample_rate':16000,
    'max_time_steps':2000,
    'keep_prob':[.5, .5, .5, .5, .5, .5, .5, .5, .5, .5, .5],
    'num_hidden':800,
    'num_hidden_fc':11,  # 及就是多少个中文字符，简体字加上罗马字母表加上空格
    'max_char_len':100,
    'nb_epoch':1000,
    'batch_size':5,
    'check_point':'./checkout'
    }

deep_speech_2 = Struct(**baidu_deep_speech_2)
tencent_speech = Struct(**tencent_speech_dict)
classify_config = Struct(**classify_dict)