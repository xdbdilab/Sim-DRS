import predict_request as pr
import parameter
import pso
import ants_multi
import nsga2
import Sim_DRS
import object_function as of

import csv
import pandas as pd
import numpy as np
import math
import time


if __name__ == "__main__":


    for i in range(int(parameter.request_num / parameter.request_interval)):
        dataframe = pd.read_csv("data/predict_request.csv", engine="python")
        all_request = dataframe.values
        request = all_request[i * parameter.request_interval : i * parameter.request_interval + 20]

        request_median = int(np.median(request))

        request_service = request_median * parameter.s_r

        instance_num = request_service / parameter.s_q

        instance = [0, 0, 0, 0, 0]
        for j in range(instance_num.shape[1]):
            if instance_num[0, j] < 1:
                instance_num[0, j] = 1
            else:
                instance_num[0, j] = round(instance_num[0, j])
            instance[j] = int(instance_num[0, j])



        result = ants_multi.ACA_ALG(instance)
        result = result.reshape((1, 20))
        pd_result = pd.DataFrame(result)
        pd_result.to_csv("data/result_ant.csv", index=False, header=None, mode="a")

        # result = pso.pso_run(instance)
        # result = result.reshape((1, 20))
        # pd_result = pd.DataFrame(result)
        # pd_result.to_csv("data/result_pso.csv", index=False, header=None, mode="a")

        # result = nsga2.nsga2(instance)
        # result = result.reshape((1, 20))
        # pd_result = pd.DataFrame(result)
        # pd_result.to_csv("data/result_ga.csv", index=False, header=None, mode="a")

        # result = Sim_DRS.nsga2(instance)
        # result = result.reshape((1, 20))
        # pd_result = pd.DataFrame(result)
        # pd_result.to_csv("data/result_ga_kmeans.csv", index=False, header=None, mode="a")


        of.update_last_d_mat(result)
