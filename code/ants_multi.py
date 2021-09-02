import numpy as np
from numpy import random as rd
import object_function as of
np.set_printoptions(linewidth=1000, suppress=True)


def main():

    loop_n = 1
    vm_list = [10, 10, 10, 12]
    per_0 = 6
    alpha = 1
    beta = 1
    rho = 0.5
    iter_times = 100
    Q = 1
    service_list = [7, 9, 10, 8, 9]
    aca = ACA(vm_list, 0, per_0, alpha, beta, rho, Q, iter_times, service_list)

    result = []
    for i in range(loop_n):
        result.append(aca.run())


def ACA_ALG(service_list):
    loop_n = 1
    vm_list_inner = [10, 10, 10, 12]
    per_0 = 700
    alpha = 1
    beta = 1
    rho = 0.2
    iter_times = 50
    Q = 1
    aca = ACA(vm_list_inner, 0, per_0, alpha, beta, rho, Q, iter_times, service_list)
    result = aca.run()
    temp = [item["cost"]+item["distance"]+item["stability"] for item in result]

    return result[temp.index(min(temp))]["mat"]

class ACA(object):

    def __init__(self, vm_list, m, pher_0, alpha, beta, rho, Q, iter_times, service_list=[1 for i in range(5)],r_0=0.3):
        self.r_0 = r_0 
        self.iter_times = iter_times
        self.Q = Q
        self.rho = rho
        self.alpha = alpha
        self.beta = beta
        self.service_list = service_list
        self.vm_list = vm_list
        self.vm_num = len(self.vm_list) 
        self.cityNum = self.vm_num * len(self.service_list)
        self.m = 50
        self.pher_mat = np.array(
            [[pher_0] * self.cityNum] * self.cityNum,dtype=np.float)

        self.ant_init_position = rd.randint(
            0, self.cityNum, self.m)
    
        self.heu_f = self.calc_heu_f([],[])

        self.f = [of.cal_instance_distance, of.cal_dpm_cost, of.cal_vm_cost]
        self.A = []

    def run(self):
        for i in range(self.iter_times):
            passed_city = [[city] for city in self.ant_init_position]

            left_service = []
            left_vm = []
            for city in self.ant_init_position:
                sev, vm_j = self.divide_to_i_j(city)
                service = self.service_list[:]
                service[sev] = service[sev]-1
                left_service.append(service)

                vm_mat = self.vm_list[:]
                vm_mat[vm_j] = vm_mat[vm_j]-1
                left_vm.append(vm_mat)

            for j in range(sum(self.service_list)-1):
                for loop in range(self.m):
                    passed_ct = passed_city[loop]
                    left_sv = left_service[loop]
                    left_v = left_vm[loop]
                    allow_city = self.get_allow_city(passed_ct, left_sv,left_v)

                    try:
                        if len(allow_city)==0:
                            raise Exception()
                    except:
                        print("The length of allow_city is 0")
                        exit(0)

                    p = self.calc_p(passed_ct, allow_city)
                    select_city_index = self.get_select_city_index(p)
                    select_city = 0
                    try:
                        select_city = allow_city[select_city_index]
                    except:
                        print("allow_city.length:",len(allow_city))
                        print("p.length:",len(p))
                        print("index:",select_city_index)
                    passed_ct.append(select_city)
                    sev, vm_j = self.divide_to_i_j(select_city)
                    left_sv[sev] -= 1
                    left_v[vm_j] -= 1

            delta_pher_k = [] 
            delta_pher_mat = np.zeros((self.cityNum, self.cityNum))
            for ant_passed_city in passed_city:
                accumulate_lenth = self.calc_passed_lenth(ant_passed_city)
                delta_pher_k.append(self.Q * accumulate_lenth)

            for ii in range(self.m):
                for k in range(sum(self.service_list)-1):
                    delta_pher_mat[passed_city[ii][k],
                                   passed_city[ii][k + 1]] += delta_pher_k[ii]

                delta_pher_mat[passed_city[ii][-1],
                               passed_city[ii][0]] += delta_pher_k[ii]
            self.pher_mat = (1 - self.rho) * self.pher_mat + delta_pher_mat

            self.ant_init_position = rd.randint(
                0, self.cityNum, self.m)
        return self.A

    def calc_heu_f(self, passed_ct, allow_city):
        local_heu_f = np.ones((self.cityNum, self.cityNum),dtype=np.float)

        return 1/local_heu_f

    def calc_p(self, passed_ct, allow_city):
        p = []
        local_heu_f = self.heu_f[passed_ct[-1]]
        for allow_ct in allow_city:
            r = rd.randint(100)*1.0/1000
            p_dividend = self.pher_mat[passed_ct[-1], allow_ct] ** self.alpha * \
                local_heu_f[allow_ct] ** self.beta
            p_divisor = 0
            for allow_ct_inner in allow_city:
                p_divisor += self.pher_mat[passed_ct[-1], allow_ct_inner] ** self.alpha * \
                    local_heu_f[allow_ct] ** self.beta
            if p_divisor < 0.01:
                p_divisor = 1
            p_ = p_dividend / p_divisor + r 
            p.append(p_)
        return p

    def get_select_city_index(self, p):

        if rd.random() >= self.r_0:
            select_city_index = p.index(max(p))
        else:
            select_city_index = self.select(p)
        if select_city_index >= len(p):
            select_city_index = p.index(max(p))
        return select_city_index

    def select(self, p):
        index = 0
        sum_ = 0
        ran = rd.random()
        for r in p:
            sum_ += r
            if ran < sum_:
                break
            index += 1
        return index

    def calc_passed_lenth(self, passed_city,is_list=True):
        if is_list:
            return self.multi_calc_passed_lenth(self.calc_path_mat(passed_city))
        return self.multi_calc_passed_lenth(passed_city)

    def multi_calc_passed_lenth(self,d_mat):
        zhipei_code = self.zhipei(d_mat)
        if zhipei_code==-1:
            return 0
        size = len(self.A)
        if size == 1:
            return 0

        return np.min(np.sqrt([((self.A[i]['cost']-self.A[size-1]['cost'])**2 + \
                        (self.A[i]["distance"]-self.A[size-1]["distance"])**2 + \
                         (self.A[i]["stability"]-self.A[size-1]["stability"])**2)  for i in range(size-1)]))
        

    def get_allow_city(self, passed_ct, left_sv, left_vm):
        not_allow = []
        for i in range(len(left_sv)):
            if left_sv[i] == 0:
                for j in range(self.vm_num):
                    not_allow.append(self.comb_i_j(i, j))
        for j in range(len(left_vm)):
            if left_vm[j]==0:
                for i in range(len(self.service_list)):
                    not_allow.append(self.comb_i_j(i,j))
        return list(set(range(self.cityNum))-set(not_allow))

    def calc_path_mat(self, best_path):
        result_mat = np.zeros(
            (len(self.service_list), self.vm_num), dtype=int)
        for city in best_path:
            i, j = self.divide_to_i_j(city)
            result_mat[i][j] += 1
        return result_mat


    def divide_to_i_j(self, city):
        return int(city/self.vm_num), city % self.vm_num

    def comb_i_j(self, i, j):
        return i*self.vm_num+j

    def zhipei(self,d_mat):
        
        cost = self.f[2](d_mat)
        stability = self.f[1](d_mat)
        distance = self.f[0](d_mat)

        if len(self.A)==0:
            self.A.append({"mat":d_mat,"cost":cost,"stability":stability,"distance":distance})
            return 0

        result_code = 1
        i = len(self.A)-1 
        while i>=0:
            if (self.A[i]['cost'] <= cost) and \
            (self.A[i]["stability"] <= stability) and \
            (self.A[i]["distance"] <= distance):
                result_code = -1
                break
            elif (self.A[i]['cost'] >= cost) and (self.A[i]["stability"] >= stability) and (self.A[i]["distance"] >= distance):
                self.A.pop(i)
            i -= 1
        self.A.append({"mat":d_mat,"cost":cost,"stability":stability,"distance":distance})
        return result_code


if __name__ == "__main__":
    main()
