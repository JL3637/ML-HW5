import numpy as np
from libsvm.svmutil import *

E_in_list = []
SV_nr_list = []
for k in [2,3,4,5,6]:
    y, X = svm_read_problem('train.txt')
    for i in range(len(y)):
        if y[i] != k:
            y[i] = -1
        else:
            y[i] = 1
    prob  = svm_problem(y, X)
    param = svm_parameter('-t 1 -g 1 -r 1 -d 2 -c 1 -q')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y, X, model, '-q')
    E_in_list.append(1 - p_acc[0]/100)
    SV_nr_list.append(model.get_nr_sv())
print(E_in_list)
print(SV_nr_list)
