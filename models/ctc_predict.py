'''ctc decoder.
'''

import tensorflow as tf
from keras.models import load_model
import numpy as np
from keras.backend import epsilon
from data_utils import vericode_data_process
import keras.backend as K

def verification_code_decoder(model_path, data_dir, algo_config,
                              vocab=dict([(i, i) for i in range(10)])):
    """验证码预测解码处理。
    :Params
        model_path: The path of the model.
        vocab: The mapping of the model predict indexes and the tokens. 
    """
    
    """
    # inputs shape=(sample_size, time_steps, feature_num)
    # y_true shape=(sample_size, label_num)
    # label_length=(sample_size, length of label of each sample)
    
    """ 
    inputs, y_true, label_length = vericode_data_process(data_dir=data_dir, 
                                                         args=algo_config,
                                                         save_dir=None,
                                                         type=algo_config.feature_type, single=False)
    model = load_model(model_path)
    # y_pred shape=(sample_size, time_steps, num_classes)
    y_pred = model.predict(inputs, batch_size=200)
    decodes  = K.ctc_decode(y_pred, 
                            input_length=np.ones(y_pred.shape[0]) * y_pred.shape[1], 
                            greedy=True, beam_width=3, top_paths=10)[0][0]
    for i in range(y_true.shape[0]):
        out = K.get_value(decodes[i])
        print('true label:{}, predict result:{}'.format(y_true[i].tolist(), out.tolist()))
    
    
    

if __name__ == '__main__':
    from  asr_config import deep_speech_2
    verification_code_decoder(model_path='D:/asr_corpus/ctc_model/fold-2deep_speech2_epoch.025-10.73.hdf5',
                              data_dir='D:/asr_corpus/验证码',
                              algo_config=deep_speech_2)