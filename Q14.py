import numpy as np
from libsvm.svmutil import *

y, X = svm_read_problem('train.txt')
y_t, X_t = svm_read_problem('test.txt')
for i in range(len(y)):
    if y[i] != 7:
        y[i] = -1
    else:
        y[i] = 1
for i in range(len(y_t)):
    if y_t[i] != 7:
        y_t[i] = -1
    else:
        y_t[i] = 1

E_out_list = []
for k in [0.01, 0.1, 1, 10, 100]:
    prob  = svm_problem(y, X)
    param = svm_parameter(f'-t 2 -g 1 -c {k} -q')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_t, X_t, model, '-q')
    E_out_list.append(1 - p_acc[0]/100)
print(E_out_list)