import numpy as np

initial_request_num = 400

request_num = initial_request_num

n_vm = 4
n_ms = 5

w_s = np.array([[0, 4, 3, 0, 0],
                [4, 0, 3, 2, 1],
                [3, 3, 0, 4, 0],
                [0, 2, 4, 0, 3],
                [0, 1, 0, 3, 0]])


r_s = np.array([[15, 21, 70, 70, 84]])

v_c = np.array([[1, 1, 1, 2]])

s_q = np.array([[2900, 2100, 650, 550, 550]])


s_r = np.array([[14, 4, 7, 4, 7]])

request_interval = 20