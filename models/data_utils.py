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


def load_char_index(file_path):
    '''load 字符字典
    # Params
        file_path: char index file. formats: chat tab idnex.
    # Return
        chat_index_list: The char index mapping.
        index_chat_list: The index char mapping.
    '''
    data = list(open(file_path, mode='r', encoding='utf-8').readlines())
    tuple_list = [e.strip('\n').split('\t') for e in data]
    char_index_list = {k:int(v) for k, v in tuple_list}
    index_char_list = {int(v):k for k, v in tuple_list}
    return char_index_list, index_char_list

def extract_mfcc(wave_path, sample_rate=None, n_mfcc=20):
    '''抽取mfcc特征
    # 参数
        wave_path: 路径
        sample_rate: 采样率
        n_mfcc: mfcc 特征个数
    '''
    wav, sr = librosa.load(wave_path, sr=sample_rate, mono=True);
    b = librosa.feature.mfcc(wav, sr, n_mfcc=n_mfcc)
    mfcc = np.transpose(b, [1, 0])
    return mfcc


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

def thchs_data_process(data_dir,
                       char_index_dict,
                       args,
                       save_dir=None,
                       type='mfcc'):
    
    file_names = os.listdir(data_dir)
    samples = int(len(file_names) / 2)  # 样本数量
    print('目录:{}，一共有:{}个样本'.format(data_dir, samples))
    
    inputs = np.zeros(shape=(samples, args.max_time_steps, args.feature_num))
    # input_length = np.ones(samples) * (args.max_time_steps / 2)
    y_true = np.zeros(shape=(samples, args.max_char_len))
    label_length = np.zeros(shape=(samples, 1))
    counter = 0
    max_char = 0
    for name in file_names:
        if name.endswith('.wav'):
            wave_path = os.path.join(data_dir, name)
            x = None
            if type == 'mfcc':
                mfcc = extract_mfcc(wave_path, args.sample_rate, args.feature_num)
                x = expand_timeSteps(mfcc, args.max_time_steps)
            else:
                spect = extract_spectrogram(wave_path, args)
                x = expand_timeSteps(spect, args.max_time_steps)
            inputs[counter] = x
            trn_content = list(open(os.path.join(data_dir, name + '.trn'), mode='r').readlines())[0].strip('\n')
            # print(trn_content)
            trn_path = os.path.join(data_dir, trn_content)
            # print(trn_path)
            corpus = list(open(trn_path, mode='r').readlines())[0]
            corpus = ' '.join(''.join(corpus.strip('\n').split(' ')))  # 每个汉字以空格相连
            if len(corpus) > max_char:
                max_char = len(corpus)
            # corpus = ''.join(corpus.strip('\n').split(' '))
            # print(corpus)
            label_length[counter][0] = len(corpus)
            for idx, char in enumerate(corpus):
                y_true[counter][idx] = char_index_dict[char]
            counter += 1
            if counter % 100 == 0:
                print('当前处理样本数：{}'.format(counter))
    print('字符最大长度：{}'.format(max_char))
    if save_dir != None:
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
    parser = argparse.ArgumentParser(prog='thchs_preprocess',
                                     description='Script to preprocess thchs data')
    parser.add_argument("--data_dir", help="Directory of thchs dataset", type=str)
 
    parser.add_argument("--save_dir", help="Directory where preprocessed arrays are to be saved",
                        type=str)
    
    parser.add_argument("--type", help="feature types, optional:mfcc,spect. default mfcc",
                        type=str) 
    
    input_args = parser.parse_args()
    print(input_args)
    data_dir = input_args.data_dir
    save_dir = input_args.save_dir
    feature_type = input_args.type
    
    args = asr_config.deep_speech_2
    char_index_dict, index_char_dict = asr_config.char_index_dict, asr_config.index_char_dict
     
    (inputs, y_true, label_length) = \
    thchs_data_process(data_dir,
                       char_index_dict,
                       args,
                       save_dir,
                       type=feature_type)

    
