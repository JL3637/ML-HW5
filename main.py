import numpy as np
from libsvm.svmutil import *
from numpy import linalg as LA
import random

# y, X = svm_read_problem('train.txt')
# y_t, X_t = svm_read_problem('test.txt')

# y_valid
# X_valid
# y_train
# X_train

list1 = [1,2,3,4,5,6,7,8,9,10]

list2 = [[1,2],[3,4],[5,6],[7,8],[9,10],[10,11],[12,13],[14,15],[16,17],[18,19]]

tmp = list(zip(list1,list2))
print(tmp)
random.shuffle(tmp)
res1, res2 = zip(*tmp)
# res1 and res2 come out as tuples, and so must be converted to lists.
res1, res2 = list(res1), list(res2)

print(res1)
print(res2)

# E_out_list = []
# for k in [0.01, 0.1, 1, 10, 100]:
#     for i in range(len(y)):
#         if y[i] != 7:
#             y[i] = -1
#     for i in range(len(y_t)):
#         if y_t[i] != 7:
#             y_t[i] = -1
#     prob  = svm_problem(y, X)
#     param = svm_parameter(f'-t 2 -g 1 -c {k} -b 1')
#     model_ptr = libsvm.svm_train(prob, param)
#     model = toPyModel(model_ptr)
#     p_label, p_acc, p_val = svm_predict(y_t, X_t, model, '-b 1')
#     E_out_list.append(1 - p_acc[0]/100)
# print(E_out_list)






















# support_vector_coefficients = model.get_sv_coef()
# support_vectors = model.get_SV()
# nr_sv = model.get_nr_sv()
# nr_class = model.get_nr_class()
# class_labels = model.get_labels()
# sv_indices = model.get_sv_indices()
# print(support_vector_coefficients)
# print(support_vectors)
# print(nr_sv)
# print(nr_class)
# print(class_labels)
# print(sv_indices)


# w = support_vector_coefficients * support_vectors
# print(w)
# print(LA.norm(w))
