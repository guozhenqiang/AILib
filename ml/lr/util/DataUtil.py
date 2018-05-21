# -*- coding: utf-8 -*-

import json
import gzip
import numpy as np

class FeatureStatistics():
    def __init__(self, idx=-1):
        self.idx=idx
        self.min=99999999
        self.max=-99999999
        self.count=0
        self.positive=0
        self.negtive=0

    def update_statistics(self, value, is_positive):
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value
        self.count += 1
        if isinstance(is_positive, bool):
            if is_positive:
                self.positive += 1
            else:
                self.negtive += 1

    def read_mapping(self, line):
        idx, k, b=line.split('\t')
        self.idx=int(idx)
        self.k=float(k)
        self.b=float(b)

def calc_feature_statistics(sample_path):
    dict={}
    with gzip.open(sample_path, 'rt')  if sample_path[-3:]=='.gz' else open(sample_path, 'r') as f:
        for line in f:
            calc_sample_line(dict, line)
    return dict

def calc_sample_line(dict, line):
    tokens=line.split('\t')
    fea_list=tokens[2].split(" ")
    for fea in fea_list:
        id, value=fea.split(":")
        id=int(id)
        value=float(value)
        if id not in dict:
            feature = FeatureStatistics(id)
            dict[id]=feature
        feature=dict[id]
        feature.update_statistics(value, int(tokens[0])>0)

def save_features_statistics(dict, save_path):
    fea_list=[ value for value in dict.values()]
    fea_list.sort(key=lambda fea: fea.idx)
    with open(save_path, 'w') as f:
        for fs in fea_list:
            f.write('{}\t{}\t{}\t{}\t{}\t{}\n'.format(fs.idx, fs.min, fs.max, fs.count, fs.positive, fs.negtive))

def load_mapping(mapping_file_path):
    dict={}
    with open(mapping_file_path) as f:
        idx=0
        for line in f:
            fs=FeatureStatistics()
            fs.read_mapping(line)
            fs.compact_idx=idx
            idx+=1
            dict[fs.idx]=fs
    return dict, max([x for x in dict]), len(dict)


def mapping_line(line, mapping_dict, fea_size, shrink, feature_idx, label_idx):
    fea_list=[0]*fea_size
    items=line.split("\t")
    label=int(items[0])



def mapping_file(input_file_path, mapping_dict, max_idx, label_idx=0, feacture_idx):
    size=len(mapping_dict)
    with gzip.open(input_file_path, 'rt') if input_file_path[-3:]=='.gz' else open(input_file_path, 'r') as f:
        for line in f:
            label, fea_list=mapping_line(line, mapping_dict, size, shrink, feature_idx, label_idx)
            yield label, fea_list

def get_batch_data(sample_file_paths, mapping_dict, batch_size=256, lable_idx=0, feacture_idx=2):
    fea_size=len(mapping_dict)
    read_line_num=0
    for sample_file_path in sample_file_paths:
        for lable, fea_list in mapping_file(sample_file_path, mapping_dict, fea_size, label_idx, feacture_idx):





if __name__ == '__main__':
    sample_path="D:\\push_platform\\train_201805070904.gz"
    fea_dict=calc_feature_statistics(sample_path)
    save_path="D:\\push_platform\\fea_statitics"
    save_features_statistics(fea_dict, save_path)
    print('end')
    pass

