import numpy as np
import parameter as pm

last_d_mat = np.zeros((pm.n_ms, pm.n_vm))
run_time = 0
flag = False


def update_last_d_mat(d_mat):
    global last_d_mat
    d_mat = d_mat.reshape(pm.n_ms, pm.n_vm)
    last_d_mat = d_mat
    return last_d_mat


def cal_instance_distance(d_mat):
    n_vm = d_mat.shape[1]

    d_vm = np.ones((n_vm, n_vm), dtype=int) - np.eye(n_vm, dtype=int)
    d_i = 0

    for i in range(d_mat.shape[0] - 1):
        for k in range(i+1, d_mat.shape[0]):
            for j in range(d_mat.shape[1]):
                d_i += d_mat[i, j] * np.sum(d_mat[k, :] * d_vm[j, :]) * pm.w_s[i, k]
    return d_i * 0.05


def cal_vm_cost(d_mat):

    c_v = 0

    for i in range(d_mat.shape[0]):
        for j in range(d_mat.shape[1]):
            c_v += d_mat[i, j] * pm.r_s[0, i] * pm.v_c[0, j]
    return c_v * 0.01


def cal_dpm_cost(d_mat):
    global last_d_mat
    global flag


    d_mat_1 = d_mat - last_d_mat

    d_mat_1 = np.maximum(d_mat_1, 0)

    d_c = np.sum(d_mat_1)

    if flag == True:
        last_d_mat = d_mat
        flag = False

    return d_c