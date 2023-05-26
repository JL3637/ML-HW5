import numpy as np
import random
import math

def decision_stump(t_arr, u_arr):
    s_id_theta_g = []
    s, id, theta = -1, -1, -1
    INF = 100
    n = len(t_arr)
    m = 16
    g = 1
    for k in range(m):
        t_arr = t_arr[t_arr[:, k+1].argsort()]
        u_arr = u_arr[t_arr[:, k+1].argsort()]
        for i in [1, -1]:
            tmp_g = 0
            for j in range(n):
                if t_arr[j][0] != i:
                    tmp_g += u_arr[j]
            for j in range(n):
                if tmp_g < g:
                    g, s, id = tmp_g, i, k+1
                    if j > 0:
                        theta = (t_arr[j-1][k+1]+t_arr[j][k+1])/2
                    else:
                        theta = -INF
                if t_arr[j][0] == i:
                    tmp_g += u_arr[j]
                else:
                    tmp_g -= u_arr[j]
    s_id_theta_g.append(s)
    s_id_theta_g.append(id)
    s_id_theta_g.append(theta)
    s_id_theta_g.append(g)
    return s_id_theta_g

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
n = len(t_arr)
u_list = [1/n] * n
u_arr = np.array(u_list)
E_in_list = []
for t in range(10):
    s_id_theta_g = decision_stump(t_arr, u_arr)
    s, id, theta, g = s_id_theta_g[0], s_id_theta_g[1], s_id_theta_g[2], s_id_theta_g[3]
    print(s_id_theta_g)
    print(np.sum(u_arr))
    epsilon = g / np.sum(u_arr)
    scal_factor = math.sqrt((1-epsilon)/epsilon)
    t_arr = t_arr[t_arr[:, id].argsort()]
    u_arr = u_arr[t_arr[:, id].argsort()]
    wrong_data = 0
    for i in range(n):
        if s * sign(t_arr[i][id] - theta) == t_arr[i][0]:
            u_arr[i] = u_arr[i] / scal_factor
        else:
            u_arr[i] = u_arr[i] * scal_factor
            wrong_data += 1
    E_in_list.append(wrong_data/n)
print(E_in_list)