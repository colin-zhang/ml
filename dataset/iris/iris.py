#!/usr/bin/env python3

import csv
import random
import numpy as np

class IrisData(object):

    def __init__(self, file):
        self.file = file
        a = np.loadtxt(file)
        print(a)
    def data(self):
        pass


def loadDataset(filename, split, trainingSet=[], testSet=[]):
    with open(filename, "r") as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        print(type(dataset))
        print((dataset))
        for x in range(len(dataset)-1):
            print(x, dataset[x])
            # for y in range(4):
            #     dataset[x][y] = float(dataset[x][y])
            # if random.random() < split:
            #     trainingSet.append(dataset[x])
            # else:
            #     testSet.append(dataset[y])


def test():
    #iris_data = IrisData()
    with open("iris.csv", 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        print(type(reader))

    trainingSet = []
    testSet = []
    loadDataset("iris.csv", 0.5, trainingSet, testSet)


def main():
    test()

if __name__ == "__main__":
    main()
