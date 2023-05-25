import numpy as np
from libsvm.svmutil import *
from numpy import linalg as LA

y, X = svm_read_problem('train.txt')
for i in range(len(y)):
    if y[i] != 1:
        y[i] = -1
    else:
        y[i] = 1
prob  = svm_problem(y, X)
param = svm_parameter(f'-t 0 -c 1 -q')
model = svm_train(prob, param)
sv_coef = model.get_sv_coef()
sv = model.get_SV()
w = [0 for _ in range(16)]
for j,x in enumerate(sv):
    for i in range(16):
        w[i] += sv_coef[j][0] * x[i+1]
print(w)
print(LA.norm(w))