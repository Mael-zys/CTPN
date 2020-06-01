import torch
import cv2
import lib.dataset_handler
import lib.generate_gt_anchor
import lib.tag_anchor
import Net
import numpy as np
import os
import time
import random
import copy
import math
import configparser
anchor_height = [11, 16, 22, 32, 46, 66, 94, 134, 191, 273]

def cal_2pt(position, cy, h, bottom):
    x_left = float(position)
    x_right = float(position + 1)
    y_top = max((cy - (float(h) - 1) / 2.0)/16.0,0)
    y_bottom = min((cy + (float(h) - 1) / 2.0)/16.0,bottom)
    return [x_left, y_top, x_right, y_bottom] 


# 验证集评估
def val(net, criterion, batch_num, using_cuda, logger, img_list):
    cf = configparser.ConfigParser()
    cf.read('./config')  # 读取超参，是否使用GPU、迭代次数、模型保存频率，batch、学习率、epoch

    gpu_id = cf.get('global', 'gpu_id')
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id
    random_list = random.sample(img_list, batch_num)
    total_loss = 0
    total_cls_loss = 0
    total_v_reg_loss = 0
    total_o_reg_loss = 0
    start_time = time.time()
    for im in random_list:
        root, file_name = os.path.split(im)
        root, _ = os.path.split(root)
        name, _ = os.path.splitext(file_name)
        gt_name = 'gt_' + name + '.txt'
        #gt_name = name + '.txt'
        gt_path = os.path.join(root, "train_gt", gt_name)
        if not os.path.exists(gt_path):
            print('Ground truth file of image {0} not exists.'.format(gt_path))
            continue

        gt_txt = lib.dataset_handler.read_gt_file(gt_path, have_BOM=True)
        img = cv2.imread(im)
        if img is None:
            batch_num -= 1
            continue

        img, gt_txt = lib.dataset_handler.scale_img(img, gt_txt)
        tensor_img = img[np.newaxis, :, :, :]
        tensor_img = tensor_img.transpose((0, 3, 1, 2))
        if using_cuda:
            tensor_img = torch.FloatTensor(tensor_img).cuda()
        else:
            tensor_img = torch.FloatTensor(tensor_img)
        
        anchor=[]
        index_anchor=[]
        for hi in range(10):
            for j in range(int(tensor_img.shape[2]/16)):
                for k in range(int(tensor_img.shape[3]/16)):
                    anchor.append(cal_2pt(k,16.0*j+7.5,anchor_height[hi],int(tensor_img.shape[2]/16)))
                    index_anchor.append(hi)
        anchor=np.array(anchor,dtype=np.float32)
        index_anchor=np.array(index_anchor, dtype=np.int32)

        vertical_pred, score, side_refinement = net(tensor_img,anchor,index_anchor)

        IOU=np.zeros((math.ceil(tensor_img.shape[3]/16),math.ceil(tensor_img.shape[2]/16),10)).tolist()

        del tensor_img
        positive = []
        negative = []
        vertical_reg = []
        side_refinement_reg = []
        visual_img = copy.deepcopy(img)
        for ii in range(math.ceil(img.shape[1]/16)):
            for jj in range(math.ceil(img.shape[0]/16)):
                for kk in range(10):
                    IOU[ii][jj][kk]=[0]
        try:
            for box in gt_txt:
                gt_anchor, visual_img = lib.generate_gt_anchor.generate_gt_anchor(img, box, draw_img_gt=visual_img)
                IOU=lib.tag_anchor.tag_anchor(gt_anchor, score, box, IOU)
            for ii in range(math.floor(img.shape[1]/16)):
                for jj in range(math.floor(img.shape[0]/16)):
                    for kk in range(10):
                        if len(IOU[ii][jj][kk]) > 2:
                                        # vertical_reg.append((ii,jj,kk,IOU[ii][jj][kk][1],IOU[ii][jj][kk][2],IOU[ii][jj][kk][0]))
                                        # if len(IOU[ii][jj][kk]) == 4:
                                        #     #positive.append((ii,jj,kk,IOU[ii][jj][kk][0]))
                                        #     side_refinement_reg.append((ii,jj,kk,IOU[ii][jj][kk][3]))
                                        # if IOU[ii][jj][kk][0]>0.7 or len(IOU[ii][jj][kk]) == 4:
                                        #     positive.append((ii,jj,kk,IOU[ii][jj][kk][0]))
                            positive.append((ii,jj,kk,IOU[ii][jj][kk][0]))
                            vertical_reg.append((ii,jj,kk,IOU[ii][jj][kk][1],IOU[ii][jj][kk][2],IOU[ii][jj][kk][0]))
                            if len(IOU[ii][jj][kk]) == 4:
                                side_refinement_reg.append((ii,jj,kk,IOU[ii][jj][kk][3]))
                                
                        else:
                            negative.append((ii,jj,kk,IOU[ii][jj][kk][0]))
        except:
            print("warning: img %s raise error!" % im)
            batch_num -= 1
            continue

        if len(vertical_reg) == 0 or len(positive) == 0 or len(side_refinement_reg) == 0:
            batch_num -= 1
            continue

        loss, cls_loss, v_reg_loss, o_reg_loss = criterion(score, vertical_pred, side_refinement, positive,
                                                           negative, vertical_reg, side_refinement_reg)
        total_loss += float(loss)
        total_cls_loss += float(cls_loss)
        total_v_reg_loss += float(v_reg_loss)
        total_o_reg_loss += float(o_reg_loss)
    end_time = time.time()
    total_time = end_time - start_time
    print('####################  Start evaluate  ####################')
    print('loss: {0}'.format(total_loss / float(batch_num)))
    logger.info('Evaluate loss: {0}'.format(total_loss / float(batch_num)))

    print('classification loss: {0}'.format(total_cls_loss / float(batch_num)))
    logger.info('Evaluate vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_num)))

    print('vertical regression loss: {0}'.format(total_v_reg_loss / float(batch_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))

    print('side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))
    logger.info('Evaluate side-refinement regression loss: {0}'.format(total_o_reg_loss / float(batch_num)))

    print('{1} iterations for {0} seconds.'.format(total_time, batch_num))
    print('#####################  Evaluate end  #####################')
    print('\n')
    return total_loss
