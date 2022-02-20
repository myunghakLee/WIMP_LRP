# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
import json
import os
import torch.optim as optim

import os
import math
import copy


from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch
from src.data.argoverse_datamodule import ArgoverseDataModule
from src.data.argoverse_dataset import ArgoverseDataset
from src.data.dummy_datamodule import DummyDataModule
from src.models.WIMP_lrp import WIMP
from argoverse.map_representation.map_api import ArgoverseMap

import XAI_utils

import torch.backends.cudnn as cudnn
import random
from argparse import ArgumentParser

from sklearn.preprocessing import minmax_scale

parser = ArgumentParser()


os.environ["CUDA_VISIBLE_DEVICES"]="2"

import time

args = {"IFC":True, "add_centerline":False, "attention_heads":4, "batch_norm":False, "batch_size":100, "check_val_every_n_epoch":3, 
          "dataroot":'./data/argoverse_processed_simple', "dataset":'argoverse', "distributed_backend":'ddp', "dropout":0.0, 
          "early_stop_threshold":5, "experiment_name":'example', "gpus":3, "gradient_clipping":True, "graph_iter":1, 
          "hidden_dim":512, "hidden_key_generator":True, "hidden_transform":False, "input_dim":2, "k_value_threshold":10, 
          "k_values":[6, 5, 4, 3, 2, 1], "lr":0.0001, "map_features":False, "max_epochs":200, "mode":'train', "model_name":'WIMP', 
          "no_heuristic":False, "non_linearity":'relu', "num_layers":4, "num_mixtures":6, "num_nodes":1, "output_conv":True, "output_dim":2, 
          "output_prediction":True, "precision":32, "predict_delta":False, "resume_from_checkpoint":None, 
          "scheduler_step_size":[60, 90, 120, 150, 180], "seed":None, "segment_CL":False, "segment_CL_Encoder":False, 
          "segment_CL_Encoder_Gaussian":False, "segment_CL_Encoder_Gaussian_Prob":False, "segment_CL_Encoder_Prob":True, 
          "segment_CL_Gaussian_Prob":False, "segment_CL_Prob":False, "use_centerline_features":True, "use_oracle":False, "waypoint_step":5, 
          "weight_decay":0.0, "workers":8, "wta":False, "draw_image" : False, "remove_high_related_score" : True, "maximum_delete_num" : 3, 
          "save_json": False, "make_submit_file" : False, "use_hidden_feature" : True, "is_LRP": True, "adjacency_exp" : True}

for k in args:
    parser.add_argument(str("--" + k), default = args[k], type= type(args[k]))
parser.add_argument("--XAI_lambda", default = 0.7, type= float)
parser.add_argument("--name", default = "", type=str)

try:  # terminal
    parser = parser.parse_args()
except:  # Jupyter notebook
    parser = parser.parse_args(args=[])

now = time.localtime()

start_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
start_time = parser.name + str(parser.XAI_lambda) + start_time
save_folder = "results_XAI_latent/" + start_time + "___" + str(parser.XAI_lambda).replace(".", "_") + "/"

os.mkdir(save_folder)

# dataset 받아오기  loader -> dataset 

train_loader = ArgoverseDataset(parser.dataroot, mode='train', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

val_loader = ArgoverseDataset(parser.dataroot, mode='val', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

# test_loader = ArgoverseDataset(parser.dataroot, mode='test', delta=parser.predict_delta,
#                               map_features_flag=parser.map_features,
#                               social_features_flag=True, heuristic=(not parser.no_heuristic),
#                               ifc=parser.IFC, is_oracle=parser.use_oracle)


# daset  -> 로더로 올리기
train_dataset = DataLoader(train_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=True, drop_last=True)

val_dataset = DataLoader(val_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)

# test_dataset = DataLoader(test_loader, batch_size=parser.batch_size, num_workers=parser.workers,
#                                 pin_memory=True, collate_fn=ArgoverseDataset.collate,
#                                 shuffle=False, drop_last=False)


# strict 
model = WIMP(parser) # _p 파라미터 추가한 모델 (for XAI ) 
# 모델 로더  : stric=False : # 학습할 때에는 graph 모듈에서 p에 해당하는 network가 없었으므로
model.load_state_dict(torch.load("experiments/example_old/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) 

# model =nn.parallel.DataParallel(model)

model = model.cuda()

# 확인 -> 웨이트 제대로 가져 왔는지 웨이트는 유지되야함 
# conv_weight_origin = copy.deepcopy(model.module.decoder.xy_conv_filters[0].weight)
# last_weight_origin = copy.deepcopy(model.module.decoder.value_generator.weight)
conv_weight_origin = copy.deepcopy(model.decoder.xy_conv_filters[0].weight)
last_weight_origin = copy.deepcopy(model.decoder.value_generator.weight)


# ADE / FDE 계산시 남은 신의 갯수 = length, 추후 배치 내 남은 갯수의 ADE / FDE 값으로  avg
def get_metric(metric_dict, ade,fde,mr,loss, length):
    metric_dict["ade"] += (ade * length).cpu().item()
    metric_dict["fde"] += (fde * length).cpu().item()
    metric_dict["mr"] += (mr * length).cpu().item()
    metric_dict["loss"] += (loss * length).cpu().item()
    metric_dict["length"]+=length


#  optimizer는 선언하지 않았습니다.
#  따라서 아마 model weight에 gradient는 계속 쌓이겠지만 저희가 중요한 것은 adjacency matrix의 graident이므로 상관 없을 것으로 예측됩니다.

slicing_2Dpadding = lambda a, idx: torch.cat((a[:idx], a[idx+1:], torch.zeros_like(a[0:1])), axis=0)
slicing_1Dpadding = lambda a, idx: F.pad(torch.cat((a[:idx], a[idx+1:]), axis=0), (0,1))

abs_min = lambda weight, k: torch.topk(abs(weight)[1:], k+1, largest = False).indices[k].item()
abs_max = lambda weight, k: torch.topk(abs(weight)[1:], k+1).indices[k].item()
simple_min = lambda weight, k: torch.topk(weight[1:], k+1, largest = False).indices[k].item()
simple_max = lambda weight, k: torch.topk(weight[1:], k+1).indices[k].item()

num_socials = []
names = ["abs_min", "abs_max", "simple_min", "simple_max"]


write_json_original = [] # ade / fde / mr / loss 저장
write_json_delete = [[[],[],[]] , [[],[],[]], [[],[],[]], [[],[],[]]]  # abs min / abs max / simple min  / simple max / 3 개 까지만 지우니까 


import pickle

for dataset in [val_dataset, train_dataset]:
    
    for batch_idx, batch in enumerate(tqdm(dataset)):

        input_dict, target_dict = batch[0], batch[1]

        # get cuda 
        input_dict["agent_features"] = input_dict["agent_features"].cuda()
        input_dict["social_features"] = input_dict["social_features"].cuda()
        input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
        input_dict["adjacency"] = input_dict["adjacency"].cuda()
        input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
        input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()


        # all_dist_params

        
        # agent_trj
        # social_agents_trj
        # agent_oracle_centerline
        # social_oracle_centerline

        # gan_features  : encorder output / GAT input : batch x number of nodes x 1 x 512
        # attention
        # adjacency
        # graph_output  : GAT  output  /  decorder innput : batch x 1 x 512

        # pred_trj
        # pred_way

        input_dict["agent_features"].requires_grad = True
        input_dict["social_features"].requires_grad = True
        input_dict["ifc_helpers"]["agent_oracle_centerline"].requires_grad = True
        input_dict["ifc_helpers"]["social_oracle_centerline"].requires_grad = True


        preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output, graph_output_s = model(**input_dict)
        
        adjacency.retain_grad()
        gan_features.retain_grad()
        graph_output.retain_grad()
        graph_output_s.retain_grad()

        input_dict["agent_features"].retain_grad()
        input_dict["social_features"].retain_grad()
        input_dict["ifc_helpers"]["agent_oracle_centerline"].retain_grad()
        input_dict["ifc_helpers"]["social_oracle_centerline"].retain_grad()
        
        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        loss.backward()
        
        graph_output_s_lrp = (graph_output_s.grad * graph_output_s).cpu().detach()
        adjacency_lrp = (adjacency.grad * adjacency).cpu().detach()
        graph_output_lrp = (graph_output.grad * graph_output).cpu().detach()
        gan_features_lrp = (gan_features.grad * gan_features).cpu().detach()
        agent_features_lrp = (input_dict["agent_features"].grad * input_dict["agent_features"]).cpu().detach()
        social_features_lrp = (input_dict["social_features"].grad * input_dict["social_features"]).cpu().detach()
        agent_oracle_centerline_lrp = (input_dict["ifc_helpers"]["agent_oracle_centerline"].grad * input_dict["ifc_helpers"]["agent_oracle_centerline"]).cpu().detach()
        social_oracle_centerline_lrp = (input_dict["ifc_helpers"]["social_oracle_centerline"].grad * input_dict["ifc_helpers"]["social_oracle_centerline"]).cpu().detach()

        for it in range(len(input_dict["agent_features"])): # batch수만큼 반복문을 돌림
            write_dict = {}
            write_dict['file_path'] = input_dict['ifc_helpers']['file_path'][it]
            write_dict['adjacency_lrp'] = adjacency_lrp[it].numpy()
            write_dict['gan_features_lrp'] = gan_features_lrp[it].numpy()
            write_dict['agent_features_lrp'] = agent_features_lrp[it].numpy()
            write_dict['social_features_lrp'] = social_features_lrp[it].numpy()
            write_dict['agent_oracle_centerline_lrp'] = agent_oracle_centerline_lrp[it].numpy()
            write_dict['social_oracle_centerline_lrp'] = social_oracle_centerline_lrp[it].numpy()
            write_dict['graph_output_lrp'] = graph_output_lrp[it].numpy()
            write_dict['graph_output_s_lrp'] = graph_output_s_lrp[it].numpy()
            write_dict['metric'] = {
                'fde' : float(fde[it]),
                'ade': float(ade[it]),
                'mr' :  float(mr[it])
            }
            save_path = write_dict['file_path'].replace("argoverse_processed_simple", "argoverse_with_LRP")
            with open(save_path, 'wb') as f:
                pickle.dump(write_dict, f)

graph_output_s.grad.shape


