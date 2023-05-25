import numpy as np
from libsvm.svmutil import *
import random
from tqdm import tqdm

y, X = svm_read_problem('train.txt')
for i in range(len(y)):
    if y[i] != 7:
        y[i] = -1
    else:
        y[i] = 1

cnt_list = [0, 0, 0, 0, 0]
for z in tqdm(range(500)):
    tmp = list(zip(y, X))
    random.shuffle(tmp)
    y_tmp, X_tmp = zip(*tmp)
    y_tmp, X_tmp = list(y_tmp), list(X_tmp)
    y_valid, X_valid = y_tmp[:200], X_tmp[:200]
    y_train, X_train = y_tmp[200:], X_tmp[200:]
    E_val_list = []
    for k in [0.1, 1, 10, 100, 1000]:
        prob  = svm_problem(y_train, X_train)
        param = svm_parameter(f'-t 2 -g {k} -c 0.1 -q')
        model_ptr = libsvm.svm_train(prob, param)
        model = toPyModel(model_ptr)
        p_label, p_acc, p_val = svm_predict(y_valid, X_valid, model, '-q')
        E_val_list.append(1 - p_acc[0]/100)
    min = 1
    id = 0
    for k in range(5):
        if(E_val_list[k] < min):
            min = E_val_list[k]
            id = k
    cnt_list[id] += 1
    print(z)
    print(min)
    print(cnt_list)
print(f'done {cnt_list}')