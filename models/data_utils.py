'''
Created on 2017年12月21日

@author: yunbin.li

The file is wave process

'''

import argparse
import os

import librosa

import asr_config
import numpy as np
import glob
import scipy.io.wavfile as wav
from python_speech_features import mfcc


def extract_mfcc(wave_path, args):
    '''抽取mfcc特征
    # 参数
        wave_path: 路径
        sample_rate: 采样率
        n_mfcc: mfcc 特征个数
    '''
    fs, audio = wav.read(wave_path)
    feature = mfcc(signal=audio, samplerate=fs, 
                   winlen=args.window_size, winstep=args.window_stride, 
                   numcep=args.feature_num, winfunc=np.hamming)
    return feature
#     wav, sr = librosa.load(wave_path, sr=sample_rate, mono=True);
#     b = librosa.feature.mfcc(wav, sr, n_mfcc=n_mfcc)
#     mfcc = np.transpose(b, [1, 0])
#     return mfcc


def extract_spectrogram(wave_path, args):
    '''
    #Return
        [timeSteps, feature_num]
    '''
    n_fft = int(args.sample_rate * args.window_size)
    win_length = n_fft
    hop_length = int(args.sample_rate * args.window_stride)
    
    # STFT
    wav, sr = librosa.load(wave_path, sr=args.sample_rate, duration=8, mono=True);
    # wav = wav[:16000 * 8]
    D = librosa.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_length, window=args.window)
    spect, phase = librosa.magphase(D)
    # S = log(S+1)
    spect = np.log1p(spect)
    
    if args.feature_normalize:
        mean = spect.mean()
        std = spect.std()
        spect = spect - mean
        spect = spect / std
    spect = np.transpose(spect, [1, 0])
    return spect
        
    
def expand_timeSteps(orgin_data, max_time_steps):
    '''扩展 mfcc特征 时间维度,通过补0操作
    # 参数
        orgin_data: [time_steps,n_mfcc]
    # 返回
        [max_time_steps,n_mfcc]
    '''
    orgin_shape = orgin_data.shape
    data = np.concatenate((orgin_data, np.zeros(shape=(max_time_steps - orgin_shape[0], orgin_shape[1]))), axis=0)
    return data

def vericode_data_process(data_dir, args, save_dir, type='mfcc'):
    """verication corpus preprocess.
    
    """
    files = glob.glob(data_dir+'/*.wav', recursive=True)
    print('total sample number are:{}'.format(len(files)))
    samples = len(files)
    # 声学特征
    import pdb
    pdb.set_trace()
    inputs = np.zeros(shape=(samples, args.max_time_steps, args.feature_num), dtype='float32')
    
    # 语音对应的验证码ID
    y_true = np.zeros(shape=(samples, args.max_char_len), dtype='int32')    
    
    # 每个语音样本对应文字的token数量，验证码固定为4
    label_length = np.ones(shape=(samples, 1), dtype='int32') * 4
    
    for idx, wave_path in enumerate(files):
        x = None
        if type == 'mfcc':
            mfcc_feature = extract_mfcc(wave_path, args)
            print(mfcc_feature.shape)
            x = expand_timeSteps(mfcc_feature, args.max_time_steps)
        else:
            spect = extract_spectrogram(wave_path, args)
            print(spect.shape)
            x = expand_timeSteps(spect, args.max_time_steps)
        print(os.path.basename(wave_path))
        trn_content = os.path.basename(wave_path).split('.')[0].split('+')[1]
        print('trn_content is:{}'.format(trn_content))
        inputs[idx] = x
        # 0~9 编码为 0~9
        for index, num_str in  enumerate(trn_content):
            y_true[idx][index] = int(num_str)
    if save_dir != None:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        np.save(save_dir + '/inputs.npy', inputs)
        np.save(save_dir + '/y_true.npy', y_true)
        np.save(save_dir + '/label_length.npy', label_length)
    return inputs, y_true, label_length

    


def load_data(data_dir, final_timeSteps):
    
    inputs = np.load(data_dir + '/inputs.npy')
    # input_length = np.load(data_dir + '/input_length.npy')
    input_length = np.ones(inputs.shape[0]) * final_timeSteps
    y_true = np.load(data_dir + '/y_true.npy')
    label_length = np.load(data_dir + '/label_length.npy')
    return inputs, input_length, y_true, label_length


if __name__ == '__main__':
    deep_speech2_config = asr_config.deep_speech_2
    vericode_data_process('../data', deep_speech2_config, '../features', type=deep_speech2_config.feature_type)
#     parser = argparse.ArgumentParser(prog='thchs_preprocess',
#                                      description='Script to preprocess thchs data')
#     parser.add_argument("--data_dir", help="Directory of thchs dataset", type=str)
#  
#     parser.add_argument("--save_dir", help="Directory where preprocessed arrays are to be saved",
#                         type=str)
#     
#     parser.add_argument("--type", help="feature types, optional:mfcc,spect. default mfcc",
#                         type=str) 
#     
#     input_args = parser.parse_args()
#     print(input_args)
#     data_dir = input_args.data_dir
#     save_dir = input_args.save_dir
#     feature_type = input_args.type
#     
#     args = asr_config.deep_speech_2
#     char_index_dict, index_char_dict = asr_config.char_index_dict, asr_config.index_char_dict
#      
#     (inputs, y_true, label_length) = \
#     thchs_data_process(data_dir,
#                        char_index_dict,
#                        args,
#                        save_dir,
#                        type=feature_type)

    
