# Homework 5
1. (d)<br>
2. (c)<br>
3. (a)<br>
4. (e) ?<br>
5. (b)<br>
6. (b)<br>
7. (c)<br>
8. (e)<br>
9. (d)<br>
10. (e) ?<br>
11. (c) 6.309673609961579<br>
12. (d) [0.011333333333333306, 0.006761904761904747, 0.009619047619047638, 0.014857142857142902, 0.01123809523809527]<br>
13. (b) [588, 368, 499, 642, 503]<br>
14. (d) [0.04519999999999991, 0.04519999999999991, 0.01419999999999999, 0.0040000000000000036, 0.00539999999999996]<br>
15. (c) [0.04519999999999991, 0.04519999999999991, 0.040200000000000014, 0.04519999999999991, 0.04519999999999991]<br>
16. (a) [320, 0, 180, 0, 0]<br>
17. (a) 0.09846547314578005<br>
18. (c) 0.571611253196931<br>
19. (a) 0.0<br>
20. (a) 0.002793296089385475<br>

code:<br>
Q11
```Python
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
```
Q12~13
```Python
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
```
Q14
```Python
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
```
Q15
```Python
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
for k in [0.1, 1, 10, 100, 1000]:
    prob  = svm_problem(y, X)
    param = svm_parameter(f'-t 2 -g {k} -c 0.1 -q')
    model = svm_train(prob, param)
    p_label, p_acc, p_val = svm_predict(y_t, X_t, model, '-q')
    E_out_list.append(1 - p_acc[0]/100)
print(E_out_list)
```
Q16
```Python
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
```
Q17~20
```Python
import numpy as np
import random
import math
from tqdm import tqdm

def decision_stump(t_arr, u_arr):
    s_id_theta = []
    s, id, theta = -1, -1, -1
    INF = 10000
    n = len(t_arr)
    m = 16
    g = 1
    for k in range(m):
        z = t_arr[:, k+1].argsort()
        t_arr = t_arr[z]
        u_arr = u_arr[z]
        for i in [1, -1]:
            tmp_g = 0
            for j in range(n):
                if t_arr[j][0] != i:
                    tmp_g += u_arr[j]
            if tmp_g < g:
                g, s, id, theta = tmp_g, i, k+1, -INF
            for j in range(n-1):
                if t_arr[j][0] == i:
                    tmp_g += u_arr[j]
                else:
                    tmp_g -= u_arr[j]
                if tmp_g < g and t_arr[j][k+1] != t_arr[j+1][k+1]:
                    g, s, id = tmp_g, i, k+1
                    theta = (t_arr[j][k+1]+t_arr[j+1][k+1])/2
    s_id_theta.append(s)
    s_id_theta.append(id)
    s_id_theta.append(theta)
    return s_id_theta

def sign(a):
    if a >= 0:
        return 1
    else:
        return -1

train_list, test_list = [], []

with open('train.txt') as file:
    for line in file:
        line = line.strip().split()
        if line[0] == '11':
            tmp = [1]
            for i in range(16):
                line2 = line[i+1].split(':')
                tmp.append(float(line2[1]))
            train_list.append(tmp)
        elif line[0] == '26':
            tmp = [-1]
            for i in range(16):
                line2 = line[i+1].split(':')
                tmp.append(float(line2[1]))
            train_list.append(tmp)

with open('test.txt') as file:
    for line in file:
        line = line.strip().split()
        if line[0] == '11':
            tmp = [1]
            for i in range(16):
                line2 = line[i+1].split(':')
                tmp.append(float(line2[1]))
            test_list.append(tmp)
        elif line[0] == '26':
            tmp = [-1]
            for i in range(16):
                line2 = line[i+1].split(':')
                tmp.append(float(line2[1]))
            test_list.append(tmp)

t_arr = np.array(train_list)
tt_arr = np.array(train_list)
test_arr = np.array(test_list)
n = len(t_arr)
nn = len(test_list)
u_list = [1/n] * n
u_arr = np.array(u_list)
E_in_list = []
Ein_G_list = [0]*n
Eout_G_list = [0]*nn
for t in tqdm(range(1000)):
    s_id_theta = decision_stump(t_arr, u_arr)
    s, id, theta = s_id_theta[0], s_id_theta[1], s_id_theta[2]
    z = t_arr[:, id].argsort()
    t_arr = t_arr[z]
    u_arr = u_arr[z]
    wrong_data = 0
    epsilon = 0
    for i in range(n):
        if s * sign(t_arr[i][id] - theta) != t_arr[i][0]:
            wrong_data += 1
            epsilon += u_arr[i]
    epsilon = epsilon / sum(u_arr)
    scal_factor = math.sqrt((1-epsilon)/epsilon)
    for i in range(n):
        if s * sign(t_arr[i][id] - theta) == t_arr[i][0]:
            u_arr[i] = u_arr[i] / scal_factor
        else:
            u_arr[i] = u_arr[i] * scal_factor
    E_in_list.append(wrong_data/n)
    a_t = np.log(scal_factor)
    for i in range(n):
        Ein_G_list[i] += a_t * (s * sign(tt_arr[i][id] - theta))
    for i in range(nn):
        Eout_G_list[i] += a_t * (s * sign(test_arr[i][id] - theta))

for i in range(n):
    Ein_G_list[i] = sign(Ein_G_list[i])
Ein_G = 0
for i in range(n):
    if Ein_G_list[i] != tt_arr[i][0]:
        Ein_G += 1
Ein_G = Ein_G / n

for i in range(nn):
    Eout_G_list[i] = sign(Eout_G_list[i])
Eout_G = 0
for i in range(nn):
    if Eout_G_list[i] != test_arr[i][0]:
        Eout_G += 1
Eout_G = Eout_G / nn

print(min(E_in_list))
print(max(E_in_list))
print(Ein_G)
print(Eout_G)
```
