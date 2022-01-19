# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
import numpy as np
import torch
import math
import copy

# +
from argoverse.map_representation.map_api import ArgoverseMap

am = ArgoverseMap()


# -

def walks(A):
    w = []
    
    for v1 in torch.arange(len(A)):
        for v2 in torch.where(A[v1])[0]:
            for v3 in torch.where(A[v2])[0]:
                w.append([v1, v2, v3])
    return torch.FloatTensor(w).cuda()


def denormalization(arr, angle, translation_x, translation_y): # 테스트 결과근 normalize되서 나오므로 이를 다시 denormalization 시키는 코드

    theta = (angle)/180*math.pi
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c, -s), (s, c)))
    
    #rotate
    arr = np.array([R.dot(arr[...,:2][i].reshape((2,1))).flatten() for i in range(len(arr[...,:2]))])

    #translate
    arr[...,0] += translation_x
    arr[...,1] += translation_y
    return arr


def draw_attention(agent_features, social_features, preds, city_name, rotation, translation, weight, head_num = 0, draw_future = True, figsize = (8,8), save_fig = False, save_name = None):

    plt.ion()
    plt.figure(figsize=figsize)
    weight = weight[head_num]
    weight_original = copy.deepcopy(weight)
#     print(weight.shape)
    minimum = np.min(weight[0, :]) 
    maximum = np.max(weight[0, :])
    plt.title("Min Weight: %.3f  Max Weight: %.3f" %(minimum, maximum))

    weight -= np.min(weight[0, :])    
    max_weight = max(np.max(weight[0, :]), 0.00001)
    weight /= max_weight

    
    angle_ans = -rotation
    xmin, xmax, ymin,ymax = [9999], [-9999], [9999], [-9999]
    preds = [[preds[i] - preds[i][:1] for i in range(len(preds))]]    # calculate x's mean and y's mean in predict
    
    agent_features = denormalization(agent_features, angle_ans, -translation[0], -translation[1])
    plt.plot(agent_features[...,0],agent_features[...,1], color='blue')
    plt.scatter(agent_features[...,0][-1],agent_features[...,1][-1], color='blue',linewidth=8)
#     plt.scatter(agent_features[...,0][19],agent_features[...,1][19], color='blue')
    plt.text(agent_features[...,0][-1],agent_features[...,1][-1], np.round(weight_original[0][0], 3))

    if draw_future:
        for p in preds[0]:
            p = denormalization(p, angle_ans, -translation[0], -translation[1])
            p += agent_features[-1] - agent_features[0]
            plt.plot(p[...,0], p[...,1], linestyle='dashed', alpha = 0.5, zorder = -1)
            plt.scatter(p[...,0][-1], p[...,1][-1], linewidth=2, alpha = 0.5, zorder = -1)
            xmin, xmax = min(np.append(xmin, p[...,0])), max(np.append(xmax, p[...,0]))
            ymin, ymax = min(np.append(ymin, p[...,1])), max(np.append(ymax, p[...,1]))
    
    for i, AV in enumerate(social_features):
        AV = denormalization(np.array(AV), angle_ans, -translation[0], -translation[1])
#         if True:
        if agent_features[:,0][0] != AV[:,0][0]: # 간혹 AV의 trajectory가 0으로 초기화 되어 있는 경우가 있는데 이 경우 AGENT의 초기값과 같아져 발생하는 오류때문에 넣어줌
            AV += AV[-1] - AV[0]
            plt.plot(AV[...,0],AV[...,1], color='black')
            plt.scatter(AV[...,0][-1],AV[...,1][-1], color='black')
            plt.scatter(AV[...,0][-1],AV[...,1][-1], color='red', linewidth = 8, alpha = weight[0][i+1])
            plt.text(AV[...,0][-1],AV[...,1][-1], np.round(weight_original[0][i+1], 3))
#             print(weight[i+1])

            xmin = min(np.append(np.append(agent_features[...,0], AV[...,0]),xmin))
            xmax = max(np.append(np.append(agent_features[...,0], AV[...,0]),xmax))
            ymin = min(np.append(np.append(agent_features[...,1], AV[...,1]),ymin))
            ymax = max(np.append(np.append(agent_features[...,1], AV[...,1]),ymax))
    local_lane_polygons = am.find_local_lane_polygons([xmin-10, xmax+10, ymin-10, ymax+10], city_name)
#     print(xmin,xmax)
    for l in local_lane_polygons:
        plt.plot(l[...,0],l[...,1], linewidth='0.5', color='gray')
    
    if save_fig:
        assert save_name, "For save figure save_name is required "
        plt.savefig(save_name)
    else:
        plt.show()
    plt.close()


# +
def draw(agent_features, social_features, preds, city_name, rotation, translation, weight = None, draw_future = True, figsize = (8,8), save_fig = False, save_name = None):

    plt.ion()
    plt.figure(figsize=figsize)
    weight_original = copy.deepcopy(weight)
    

#     weight -= np.min(weight)
    max_weight = max(np.max(abs(weight)), 0.000001)
    weight /= max_weight
    
    plt.title("Max Weight: %.3f" %max_weight)
    
    angle_ans = -rotation
    xmin, xmax, ymin,ymax = [9999], [-9999], [9999], [-9999]
    preds = [[preds[i] - preds[i][:1] for i in range(len(preds))]]    # calculate x's mean and y's mean in predict
    
    agent_features = denormalization(agent_features, angle_ans, -translation[0], -translation[1])
    plt.plot(agent_features[...,0],agent_features[...,1], color='blue')
    plt.scatter(agent_features[...,0][-1],agent_features[...,1][-1], color='blue',linewidth=8)
#     plt.scatter(agent_features[...,0][19],agent_features[...,1][19], color='green')
    plt.text(agent_features[...,0][-1],agent_features[...,1][-1], np.round(weight_original[0], 3))

    if draw_future:
        for p in preds[0]:
            p = denormalization(p, angle_ans, -translation[0], -translation[1])
            p += agent_features[-1] - agent_features[0]
            plt.plot(p[...,0], p[...,1], linestyle='dashed', alpha = 0.5, zorder = -1)
            plt.scatter(p[...,0][-1], p[...,1][-1], linewidth=2, alpha = 0.5, zorder = -1)
            xmin, xmax = min(np.append(xmin, p[...,0])), max(np.append(xmax, p[...,0]))
            ymin, ymax = min(np.append(ymin, p[...,1])), max(np.append(ymax, p[...,1]))
    
    for i, AV in enumerate(social_features):
        AV = denormalization(np.array(AV), angle_ans, -translation[0], -translation[1])
#         if True:
        if agent_features[:,0][0] != AV[:,0][0]: # 간혹 AV의 trajectory가 0으로 초기화 되어 있는 경우가 있는데 이 경우 AGENT의 초기값과 같아져 발생하는 오류때문에 넣어줌
            AV += AV[-1] - AV[0]
            plt.plot(AV[...,0],AV[...,1], color='black')
            plt.scatter(AV[...,0][-1],AV[...,1][-1], color='black')
            if weight[i+1] > 0:
                plt.scatter(AV[...,0][-1],AV[...,1][-1], color='red', linewidth = 8, alpha = weight[i+1])
            else:
                plt.scatter(AV[...,0][-1],AV[...,1][-1], color='green', linewidth = 8, alpha = -weight[i+1])
            plt.text(AV[...,0][-1],AV[...,1][-1], np.round(weight_original[i+1], 3))
            xmin = min(np.append(np.append(agent_features[...,0], AV[...,0]),xmin))
            xmax = max(np.append(np.append(agent_features[...,0], AV[...,0]),xmax))
            ymin = min(np.append(np.append(agent_features[...,1], AV[...,1]),ymin))
            ymax = max(np.append(np.append(agent_features[...,1], AV[...,1]),ymax))
    local_lane_polygons = am.find_local_lane_polygons([xmin, xmax, ymin, ymax], city_name)
#     print(xmin,xmax)
    for l in local_lane_polygons:
        plt.plot(l[...,0],l[...,1], linewidth='0.5', color='gray')

    if save_fig:
        assert save_name, "For save figure save_name is required "
        plt.savefig(save_name)
    else:
        plt.show()
    plt.close()        
