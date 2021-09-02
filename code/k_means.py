import numpy as np
import pandas as pd

random_num = 0
def loadDataSet(filename):
    dataframe = pd.read_csv(filename, engine="python", header=None)
    dataMat = dataframe.values
    return dataMat


def distEclud(vecA, vecB):
    return np.sqrt(np.power(vecA - vecB, 2).sum())


def randCent(dataSet, k):

    n = np.shape(dataSet)[1]

    centroids = np.mat(np.zeros((k, n)))
    for j in range(n):
        minJ = min(dataSet[:, j])
        maxJ = max(dataSet[:, j])
        rangeJ = float(maxJ - minJ)
        centroids[:, j] = minJ + rangeJ * np.random.rand(k, 1)
    return centroids


def kMeans(dataSet, k, distMeans = distEclud, creatCent = randCent):

    m = np.shape(dataSet)[0]

    clusterAssment = np.mat(np.zeros((m, 2)))

    centroids = creatCent(dataSet, k)

    clusterChanged = True

    while clusterChanged:
        clusterChanged = False

        for i in range(m):
            minDist = np.inf
            minIndex = -1

            for j in range(k):
                distJI = distMeans(centroids[j, :], dataSet[i, :])

                if distJI < minDist:
                    minDist = distJI
                    minIndex = j

            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True

            clusterAssment[i, :] = minIndex, minDist**2

        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0].A == cent)[0]]
            if len(ptsInClust) != 0:
                centroids[cent, :] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment


def main(data, n):

    dataMat = loadDataSet("data/instance_num_2.csv")
    dataMat = np.row_stack((dataMat, data))

    myCentroids, clustAssing = kMeans(dataMat, 8)

    dataframe = pd.read_csv("data/result_ga_2.csv", engine="python", header=None)
    all_deployment = dataframe.values

    all_deployment_reshape = []
    for i in range(all_deployment.shape[0]):
        now_deployment = all_deployment[i]
        now_deployment = now_deployment.reshape((5, 4))
        all_deployment_reshape.append(now_deployment)

    x = int(clustAssing[-1, 0])

    same_kind = []
    same_kind_deployment = []

    for i in range(dataMat.shape[0]-1):
        if int(clustAssing[i, 0]) == x:
            same_kind.append(dataMat[i])
            same_kind_deployment.append(all_deployment_reshape[i])
    kmeans_deployment = []
    len_same_kind_deployment = len(same_kind_deployment)


    for k in range(n):

        random_num = np.random.randint(0, len_same_kind_deployment)
        difference = np.array(data) - np.array(same_kind[random_num])
        deployment = same_kind_deployment[random_num].copy()


        if (np.array(difference) == 0).all():
            kmeans_deployment.append(same_kind_deployment[random_num])
            continue

        j = 0

        while j < difference.shape[0]:
            random_location_y = np.random.randint(0, same_kind_deployment[0].shape[1])

            if deployment[j, random_location_y] + difference[j] < 0:
                continue
            deployment[j, random_location_y] = difference[j] + deployment[j, random_location_y]
            j = j + 1

        kmeans_deployment.append(deployment)

    return kmeans_deployment