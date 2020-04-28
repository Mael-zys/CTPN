import torch  # PyTorch包
import numpy as np  #numpy库
import os  # 输入输出库
import configparser  # 命令行参数解析库，python自带
import time  # 时间库，python自带
import logging  # 输出日志库，python自带
import datetime  # 时间库，python自带
import copy  
import codecs

path = '/home/zhangyangsong/OCR/ICDAR2013/test_gt'
dest = '/home/zhangyangsong/OCR/ICDAR2013/test_gt'

def read_gt_file(path, have_BOM=False):
    result = []
    if have_BOM:
        fp = codecs.open(path, 'r', 'utf-8-sig')
    else:
        fp = open(path, 'r')
    for line in fp.readlines():
        pt = line.split(',')
        if have_BOM:
            box = [int(round(float(pt[i]))) for i in range(4)]
        else:
            box = [int(round(float(pt[i]))) for i in range(4)]
        gt=[box[0],box[1],box[0],box[3],box[2],box[3],box[2],box[1]]
        result.append(gt)
    fp.close()
    return result



if __name__ == '__main__':
    l = os.listdir(path)
    for f in l:
        gt_path=os.path.join(path, f)
        gt_dest=os.path.join(dest, f)
        gt_txt = read_gt_file(gt_path,True)
        fp = open(gt_dest, 'w')
        for gt in gt_txt:
            fp.writelines(str(gt[0])+','+str(gt[1])+','+str(gt[2])+','+str(gt[3])+','+str(gt[4])+','+str(gt[5])+','+str(gt[6])+','+str(gt[7]))
            fp.writelines("\n")
        fp.close()

