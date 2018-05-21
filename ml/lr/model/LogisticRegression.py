# -*- coding: utf-8 -*-

import numpy as np
import time
import random

class LogisticRegression():
    def __init__(self):
        self.feaid_weight={}
        self.weight_list=[]
        pass

    def load_model(self, weight_path, mapping_path):
        with open(weight_path) as f:
            self.weight_list=[float(line.split("\t")[0]) for line in f]
        self.feaid_weight[0]=self.weight_list[0]
        with open(mapping_path) as f:
            idx=1
            for line in f:
                fea_list=line.split("\t")
                self.feaid_weight[int(fea_list[0])]=float(fea_list[1])*self.weight_list[idx]+float(fea_list[2])
                idx=idx+1
        print(self.feaid_weight)
        print(self.weight_list)
        pass

    def get_predict(self, fea_dict):
        x=self.feaid_weight[0]
        for k,v in fea_dict.items():
            x=x+self.feaid_weight[k]*v
        return self.sigmoid(x)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))



if __name__ == '__main__':
    pass

