import os
import numpy as np
from numpy import random as rd
import object_function as ob
import copy
import k_means


def nsga2(service_count):
    pop_size = 50
    gen_count = 50
    vm_num = 4
    nsga = NSGA2(service_count, pop_size, gen_count, vm_num)
    result = nsga.run()
    return result


class NSGA2(object):
    def __init__(self, service_list, pop_size, gen_count, vm_num, mutation_r=0.3, cross_r=0.3):
        self.service_list = service_list
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.vm_num = vm_num
        self.mutation_r = mutation_r
        self.cross_r = cross_r

    def run(self):

        kmeans_pop = 10
        random_pop = self.pop_size - kmeans_pop

        random_last_pop = [self.gene_mat(self.service_list, len(
            self.service_list), self.vm_num) for i in range(random_pop)]

        kmeans_last_pop = k_means.main(self.service_list, kmeans_pop)

        last_pop = np.row_stack((kmeans_last_pop, random_last_pop))

        for gen_i in range(self.gen_count):
            cr_pool = self.crowd_and_rank(last_pop)

            parent_selected = self.select(cr_pool)

            last_pop = self.genetic_operator(last_pop, parent_selected)

        cr_pool = self.crowd_and_rank(last_pop)

        return last_pop[int(cr_pool[0][0])]

    def poly_mutation(self, child):
        child1 = copy.deepcopy(child)
        if rd.randint(1000) * 1.0 / 1000 > self.mutation_r:
            return child
        i = rd.randint(self.vm_num)
        j = rd.randint(self.vm_num)
        if (child[i, j] == 0):
            return child
        child[i, j] = child[i, j] - 1
        h = rd.randint(self.vm_num)
        while h == j:
            h = rd.randint(self.vm_num)
        child[i, h] = child[i, h] + 1
        if self.if_false(child):
            return self.poly_mutation(child1)
        return child

    def poly_cross(self, child1, child2, times=10):
        if (times == 0):
            return child1, child2
        if times == 10 and (rd.randint(1000) * 1.0 / 1000 > self.cross_r):
            return child1, child2
        child3 = copy.deepcopy(child1)
        child4 = copy.deepcopy(child2)

        i = rd.randint(len(self.service_list))
        temp = [k for k in child1[i]]
        child1[i] = child2[i]
        child2[i] = temp
        if self.if_false(child1) or self.if_false(child2):
            return self.poly_cross(child3, child4, times - 1)
        return child1, child2

    def genetic_operator(self, last_pop, parent_selected):
        child_offspring = []

        for i in range(int(self.pop_size / 2)):
            child1 = last_pop[int(parent_selected[i][0])]
            child2 = last_pop[int(parent_selected[self.pop_size - i - 1][0])]
            child3, child4 = self.poly_cross(child1, child2)
            last_pop[int(parent_selected[i][0])] = child3
            last_pop[int(parent_selected[self.pop_size - i - 1][0])] = child4

        for i in range(self.pop_size):
            item = last_pop[int(parent_selected[i][0])]
            child_offspring.append(self.poly_mutation(copy.deepcopy(item)))

        return child_offspring

    def select(self, pool):
        V = 1
        M = 3
        parent_selected = np.zeros([self.pop_size, V + M + 3])
        rank_col = V + M + 1
        distance_col = V + M + 2
        candidate = np.random.randint(self.pop_size, size=(self.pop_size, 2))
        for i in range(self.pop_size):
            parent = candidate[i, :]
            if pool[parent[0], rank_col] < pool[parent[1], rank_col]:
                parent_selected[i, :] = pool[parent[0], :]
            elif pool[parent[0], rank_col] > pool[parent[1], rank_col]:
                parent_selected[i, :] = pool[parent[1], :]
            else:
                if pool[parent[0], distance_col] > pool[parent[1], distance_col]:
                    parent_selected[i, :] = pool[parent[0], :]
                elif pool[parent[0], distance_col] < pool[parent[1], distance_col]:
                    parent_selected[i, :] = pool[parent[1], :]
                else:
                    parent_selected[i,
                    :] = pool[parent[np.random.randint(2)], :]
        return parent_selected


    def crowd_and_rank(self, last_pop):
        front = []
        rank = 0
        f = self.func_c(last_pop)
        index = np.zeros((self.pop_size, 1))
        for i in range(self.pop_size):
            index[i, 0] = i
        c = np.column_stack((index, f))
        n_p = np.zeros(self.pop_size)
        s_p = []
        for p in range(self.pop_size):
            s_p.append(np.where(((f[p, :] - f[:, :] <= 0).all(axis=1))
                                & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0])
            n_p[p] = len(np.where(((f[p, :] - f[:, :] >= 0).all(axis=1))
                                  & (~((f[p, :] - f[:, :] == 0).all(axis=1))))[0])
        front.append(np.where(n_p == 0)[0])

        f1 = np.column_stack((c, np.zeros(self.pop_size)))
        while len(front[rank]) != 0:
            front_indiv = front[rank]
            n_p[front_indiv] = float('inf')
            f1[front_indiv, 4] = rank
            rank += 1
            for i in range(len(front_indiv)):
                temp = s_p[front_indiv[i]]
                n_p[temp] -= 1
            front.append(np.where(n_p == 0)[0])
        sorted_f1 = f1[np.lexsort(f1.T)]
        rowsindex = 0
        sorted_f1 = np.column_stack((sorted_f1, np.zeros([self.pop_size, 3])))
        for i in range(len(front) - 1):
            l_f = len(front[i])
            if l_f > 2:

                sorted_ind = np.argsort(
                    sorted_f1[rowsindex:(rowsindex + l_f), 1: 4], axis=0)
                fmin = np.zeros(3)
                fmax = np.zeros(3)
                for m in range(3):
                    fmin[m] = sorted_f1[sorted_ind[0, m] + rowsindex, 1 + m]
                    fmax[m] = sorted_f1[sorted_ind[-1, m] + rowsindex, 1 + m]
                    sorted_f1[sorted_ind[0, m] + rowsindex,
                              3 + m + 2] = float('inf')
                    sorted_f1[sorted_ind[-1, m] +
                              rowsindex, 3 + m + 2] = float('inf')
                for j in range(1, l_f - 1):
                    for m in range(3):
                        if fmax[m] - fmin[m] == 0:
                            sorted_f1[sorted_ind[j, m] + rowsindex,
                                      3 + m + 2] = float('inf')
                        else:
                            sorted_f1[sorted_ind[j, m] + rowsindex, 3 + m + 2] = (
                                                                                         sorted_f1[sorted_ind[
                                                                                                       j + 1, m] + rowsindex, 1 + m] -
                                                                                         sorted_f1[sorted_ind[
                                                                                                       j - 1, m] + rowsindex, 1 + m]) / (
                                                                                             fmax[m] - fmin[m])
            else:
                sorted_f1[rowsindex:(rowsindex + l_f),
                3 + 2:6 + 2] = float('inf')
            rowsindex += l_f
        sorted_f1 = np.column_stack(
            (sorted_f1, sorted_f1[:, 3 + 2:6 + 2].sum(axis=1)))
        chromosome_NDS_CD1 = np.column_stack((sorted_f1[:, : 4], np.zeros(
            [self.pop_size, 1]), sorted_f1[:, 4], sorted_f1[:, 6 + 2]))
        return chromosome_NDS_CD1

    def func_c(self, last_pop):

        ff = np.zeros([self.pop_size, 3])
        for i in range(self.pop_size):
            ff[i, 0] = ob.cal_instance_distance(last_pop[i])
            ff[i, 1] = ob.cal_dpm_cost(last_pop[i])
            ff[i, 2] = ob.cal_vm_cost(last_pop[i])
        return ff

    def rand_sum(self, sum, size, rand_size=1.5):
        list = [int(sum / size) for i in range(size)]
        left = sum % size
        for i in range(left):
            index = rd.randint(size)
            list[index] += 1
        if sum >= size:
            for i in range(int(size * rand_size)):
                num = rd.randint(sum / size)
                if rd.randint(10) > 7:
                    num += 1
                index = rd.randint(size)
                while list[index] < num:
                    index = rd.randint(size)
                list[index] -= num
                list[rd.randint(size)] += num
        return list

    def gene_mat(self, sum, row, column, rand_size=1.5):
        z = 0
        mat = np.zeros((row, column), dtype=np.int)
        for k in range(row):
            mat[k] = self.rand_sum(sum[k], column, rand_size)
        if self.if_false(mat):
            return self.gene_mat(sum, row, column)
        return mat

    def if_false(self, mat, vm_list=[10, 10, 10, 12]):

        sum_co = np.sum(mat, axis=0)

        return not np.all(np.array(vm_list) - sum_co >= 0)


if __name__ == "__main__":
    for i in range(10):
        nsga2([10, 7, 6, 8, 9])
