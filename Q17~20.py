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
