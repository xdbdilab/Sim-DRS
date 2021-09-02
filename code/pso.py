import numpy as np
import object_function as ob
import parameter as pm
from itertools import product
import copy

SNUM = 4


def golfuction(last_postion, position, serv, mach):
    return ob.cal_instance_distance(position), ob.cal_dpm_cost(position), ob.cal_vm_cost(position)


class Machine:

    def __init__(self, rType, rCPU):
        self.rType = rType
        self.rCPU = rCPU


class Server:

    def __init__(self, pNUM, pCPU, pCOST):
        self.pNUM = pNUM
        self.pCPU = pCPU
        self.pCOST = pCOST


class Particle:
    def GetAdd(self, serv, pb_vec, gb_vec):
        result = []
        for i in range(5):
            tempp = self.position[i]
            tempv = self.velocity[i]

            tempadd = tempp + tempv
            unfu = []
            sum = 0
            for i, n in enumerate(tempadd):
                if n > 0:
                    unfu.append([i, n])
                    continue
                sum -= n
                tempv[i] -= n
            for i in range(sum):
                print(unfu)
                j = np.random.choice(len(unfu))
                tempv[unfu[j][0]] -= 1
                unfu[j][1] -= 1
                if unfu[j][1] == 0:
                    unfu.remove(unfu[j])
            result.append(tempv)
            if tempv.sum() != 0:
                print("p{}add{}v{}".format(tempp, tempadd, tempv))
                continue

        result = np.array(result)

        position = result + np.array(self.position)
        chaju = [0 for x in range(SNUM)]
        for i in range(SNUM):
            chaju[i] = np.sum(position[:, i]) - serv[i].pNUM

        for i in range(SNUM):
            if chaju[i] > 0:
                size = chaju[i]
                for j in range(size):
                    plist = position[:, i]
                    for index, k in enumerate(plist):
                        if k > 0:
                            position[index, i] -= 1
                            result[index, i] -= 1
                            chaju[i] -= 1
                            mlist = []
                            for m in range(SNUM):
                                if chaju[m] < 0:
                                    mlist.append(m)
                            m = np.random.choice(mlist)
                            chaju[m] += 1
                            result[index, m] += 1
                            position[index, m] += 1

        return result

    def __init__(self, serv, mach):
        self.velocity = []

        self.last_position = []
        self.position = []
        self.pValue = 0.01

        self.pb_velocity = []
        self.pb_postion = []
        self.pb_value = 0.01

        self.d_mat = np.zeros((5, SNUM), dtype=int)
        for n in mach:
            m = np.random.choice(SNUM)
            self.d_mat[n.rType][m] += 1
        temp_mat = np.zeros((5, SNUM), dtype=int)
        for n in mach:
            m = np.random.choice(SNUM)
            temp_mat[n.rType][m] += 1
        self.position = self.d_mat

        self.velocity = temp_mat - self.d_mat
        self.pb_velocity = temp_mat - self.d_mat

        pb_vec = np.zeros((5, SNUM), dtype=int)
        gb_vec = np.zeros((5, SNUM), dtype=int)

        self.velocity = self.GetAdd(serv, pb_vec, gb_vec)
        self.position += self.velocity
        self.last_position = self.position
        self.pb_postion = self.d_mat
        self.pb_velocity = self.velocity
        self.pValue = golfuction(self.last_position, self.position, serv, mach)
        self.pb_value = self.pValue


class Swarm:
    def GetAdd(self, par_index, pb_vec, gb_vec):
        par = self.partices[par_index]

        result = []
        pv = 1.0 / np.sum(par.pValue)
        pb = 1.0 / np.sum(par.pb_value)
        pg = 1.0 / np.sum(self.gb_value)
        p1 = pv / (pv + pb + pg)
        p2 = (pv + pb) / (pv + pb + pg)

        for i in range(5):
            rnd = np.random.uniform()
            tempp = par.position[i]
            if rnd <= p1:
                tempv = par.velocity[i]
            elif p1 < rnd <= p2:
                tempv = pb_vec[i]
            else:
                tempv = gb_vec[i]
            tempadd = tempp + tempv
            unfu = []
            sum = 0
            for i, n in enumerate(tempadd):
                if n > 0:
                    unfu.append([i, n])
                    continue
                sum -= n
                tempv[i] -= n
            for i in range(sum):
                j = np.random.choice(len(unfu))
                tempv[unfu[j][0]] -= 1
                unfu[j][1] -= 1
                if unfu[j][1] == 0:
                    unfu.remove(unfu[j])
            result.append(tempv)
            if tempv.sum() != 0:
                print("p{}add{}v{}".format(tempp, tempadd, tempv))
                continue
        result = np.array(result)

        position = result + np.array(par.position)
        chaju = [0 for x in range(SNUM)]
        for i in range(SNUM):
            chaju[i] = np.sum(position[:, i]) - self.serv[i].pNUM
        for i in range(SNUM):
            if chaju[i] > 0:
                size = chaju[i]
                for j in range(size):
                    plist = position[:, i]
                    for index, k in enumerate(plist):
                        if k > 0:
                            position[index, i] -= 1
                            result[index, i] -= 1
                            chaju[i] -= 1
                            mlist = []
                            for m in range(SNUM):
                                if chaju[m] < 0:
                                    mlist.append(m)
                            m = np.random.choice(mlist)
                            chaju[m] += 1
                            result[index, m] += 1
                            position[index, m] += 1
        return result

    def UpdateVelocityAndPosition(self, par_index):
        par = self.partices[par_index]
        pb_vec = par.pb_postion - par.position
        gb_vec = self.gb_position - par.position
        self.partices[par_index].velocity = self.GetAdd(par_index, pb_vec, gb_vec)
        self.partices[par_index].last_position = self.partices[par_index].position
        self.partices[par_index].position += self.partices[par_index].velocity

    def CompareBetter(self, v1, v2):
        if self.findMax:
            return v1 > v2
        else:
            return v1 < v2

    def EvalueFunc(self, par_index):
        par = self.partices[par_index]
        par.pValue = golfuction(par.last_position, par.position, self.serv, self.mach)

        if self.comparep(par.pb_value, par.pValue):
            self.partices[par_index].pb_value = par.pValue
            self.partices[par_index].pb_position = par.position
            self.partices[par_index].pb_velocity = par.velocity

        temp = []
        temp.append(self.archieve[0])
        for x in self.archieve:
            k = 0
            for y in temp:
                if self.compare_eq(x.pValue, y.pValue):
                    k = k + 1
                else:
                    break
                if k == len(temp):
                    temp.append(x)
        self.archieve = temp

        i = np.random.choice(self.archieve)

        self.gb_value = i.pValue
        self.gb_position = i.position
        self.gb_velocity = i.velocity

    def compare_eq(self, fitness_curr, fitness_ref):
        for i in range(len(fitness_curr)):
            if fitness_curr[i] != fitness_ref[i]:
                return True
        return False

    def comparep(self, fitness_curr, fitness_ref):
        for i in range(len(fitness_curr)):
            if fitness_curr[i] < fitness_ref[i]:
                return False
        return self.compare_eq(fitness_curr, fitness_ref)

    def update_compare_set(self, pl):
        pl=list(set(pl))
        v = [i.pValue for i in pl]
        tf = [self.comparep(i, j) for i, j in product(v, repeat=2)]
        m = np.array(tf).reshape(len(v), len(v))
        for x in range(len(v)):
            m[x, x] = False
        ret = [i for i in range(len(v)) if not np.any(m[i, :])]
        return [pl[r] for r in ret]

    def Update_archive(self, archive_list):
        par_list=copy.deepcopy(self.partices)
        par_list = self.update_compare_set(par_list)
        par_list = np.concatenate((archive_list, par_list), axis=0)
        curr_archive_list = self.update_compare_set(par_list)
        return curr_archive_list

    def RunSwarm(self):
        size = len(self.partices)
        for curr_iter in range(self.max_iter):
            self.archieve = self.Update_archive(self.archieve)

            self.curr_iter = curr_iter
            for par_index in range(size):
                self.UpdateVelocityAndPosition(par_index)
                self.EvalueFunc(par_index)

    def __init__(self, d_mat, particle_size, max_iter, golfuction, findMax):

        self.partices = []
        self.archieve = []
        self.gb_position = 0
        self.gb_velocity = 0
        self.gb_value = 0.0

        self.serv = []
        self.mach = []
        self.particle_size = particle_size
        self.curr_iter = 0
        self.max_iter = max_iter
        self.findMax = findMax

        for stype in range(5):
            for i in range(d_mat[stype].sum()):
                self.mach.append(Machine(stype, pm.r_s[0, stype]))

        self.serv = [Server(10, 200, 1), Server(10, 200, 1), Server(10, 200, 1), Server(12, 500, 1)]
        par = Particle(self.serv, self.mach)
        par1 = Particle(self.serv, self.mach)
        self.gb_position = par.position
        self.gb_velocity = par.velocity
        self.gb_value = par.pValue
        self.partices.append(par)

        self.archieve.append(par1)
        for i in range(particle_size):
            if i == 0:
                continue
            par = Particle(self.serv, self.mach)
            par.pValue = golfuction(par.last_position, par.position, self.serv, self.mach)
            par.pb_value = par.pValue
            self.partices.append(par)
            self.EvalueFunc(i)


def pso_run(instance_num):

    instance_num = np.transpose(instance_num)
    newRows = np.zeros((5, SNUM - 1))
    d_mat = np.c_[instance_num, newRows].astype(np.int)

    findMax = False
    particle_size = 50
    max_iter = 250
    swarm = Swarm(d_mat, particle_size, max_iter, golfuction, findMax)
    swarm.RunSwarm()
    return swarm.gb_position


if __name__ == '__main__':
    instance_num = [7, 3, 4, 1, 5]
    pso_run(instance_num)
