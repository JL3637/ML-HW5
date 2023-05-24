from libsvm.svmutil import *

y, X = svm_read_problem('train.txt')
y_t, X_t = svm_read_problem('test.txt')

prob  = svm_problem(y, X)
param = svm_parameter('-t 0 -c 4 -b 1')
m = svm_train(prob, param)