'''
Created on 2017年11月24日

验证码分类模型预测代码

'''
from data_utils import  vericode_data_process
from keras.models import load_model
from asr_config import classify_config
import numpy as np
from sklearn.model_selection import KFold


def kold_predict(data_dir, algo_config, model_path):
    
    inputs, y_true, label_length = vericode_data_process(data_dir = data_dir, 
                                                         args = algo_config, 
                                                         save_dir = None, 
                                                   type=algo_config.feature_type)
    kf = KFold(n_splits=5)
    kf_num = 1
    for train_idx, validate_idx in kf.split(y_true):
        inputs_valid = inputs[validate_idx]
        y_true_valid = y_true[validate_idx]
        total_samples = inputs_valid.shape[0]
        print('predict samples:{}'.format(total_samples))
        counter = 0
        model = load_model(model_path.format(kf_num))
        probs = model.predict(inputs_valid, batch_size=200)
        pred_numbers = np.hstack([np.argsort(n, axis=1)[::, -1].reshape((n.shape[0],1)) for n in probs])
        for i in range(total_samples):
            # print('lables:{}'.format(y_true_valid[i].tolist()))
            # print('predict result:{}'.format(pred_numbers[i].tolist()))
            if y_true_valid[i].tolist() == pred_numbers[i].tolist():
                counter += 1
        print('the kfold-{}, samples:{}, accuracy:{}'.format(kf_num, total_samples, counter/float(total_samples)))
        kf_num += 1

def predict(data_dir, algo_config, model_path):
    
    inputs, y_true, label_length = vericode_data_process(data_dir = data_dir, 
                                                         args = algo_config, 
                                                         save_dir = None, 
                                                   type=algo_config.feature_type)
    total_samples = inputs.shape[0]
    print('predict samples:{}'.format(total_samples))
    counter = 0
    model = load_model(model_path)
    probs = model.predict(inputs, batch_size=200)
    pred_numbers = np.hstack([np.argsort(n, axis=1)[::, -1].reshape((n.shape[0],1)) for n in probs])
    for i in range(total_samples):
        if y_true[i].tolist() == pred_numbers[i].tolist():
            counter += 1
    print('the total samples:{}, accuracy are:{}'.format(total_samples, counter/float(total_samples)))


if __name__ == '__main__':
    predict('../test_data', classify_config, './checkout/classify_model-fold_{}.hdf5')