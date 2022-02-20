# -*- coding: utf-8 -*-
# +
# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import copy
import json
import os
import torch.optim as optim

from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch


from src.data.argoverse_datamodule import ArgoverseDataModule
from src.data.argoverse_dataset import ArgoverseDataset
from src.data.dummy_datamodule import DummyDataModule
from src.models.WIMP import WIMP
import math


import XAI_utils

import torch.backends.cudnn as cudnn
import random
os.environ["CUDA_VISIBLE_DEVICES"]= "1"

# +
args = {"IFC":True, "add_centerline":False, "attention_heads":4, "batch_norm":False, "batch_size":100, "check_val_every_n_epoch":3, 
          "dataroot":'./data/LRP_adjacency3', "dataset":'argoverse', "distributed_backend":'ddp', "dropout":0.0, 
          "early_stop_threshold":5, "experiment_name":'example', "gpus":3, "gradient_clipping":True, "graph_iter":1, 
          "hidden_dim":512, "hidden_key_generator":True, "hidden_transform":False, "input_dim":2, "k_value_threshold":10, 
          "k_values":[6, 5, 4, 3, 2, 1], "lr":0.0001, "map_features":False, "max_epochs":200, "mode":'train', "model_name":'WIMP', 
          "no_heuristic":False, "non_linearity":'relu', "num_layers":4, "num_mixtures":6, "num_nodes":1, "output_conv":True, "output_dim":2, 
          "output_prediction":True, "precision":32, "predict_delta":False, "resume_from_checkpoint":None, 
          "scheduler_step_size":[60, 90, 120, 150, 180], "seed":None, "segment_CL":False, "segment_CL_Encoder":False, 
          "segment_CL_Encoder_Gaussian":False, "segment_CL_Encoder_Gaussian_Prob":False, "segment_CL_Encoder_Prob":True, 
          "segment_CL_Gaussian_Prob":False, "segment_CL_Prob":False, "use_centerline_features":True, "use_oracle":False, "waypoint_step":5, 
          "weight_decay":0.0, "workers":8, "wta":False, "draw_image" : False, "remove_high_related_score" : False, "maximum_delete_num" : 3, 
          "save_json": False, "make_submit_file" : False, "use_hidden_feature" : True, "is_LRP": False, "adjacency_exp" : True}

from argparse import ArgumentParser
parser = ArgumentParser()


for k in args:
    parser.add_argument(str("--" + k), default = args[k], type= type(args[k]))
parser.add_argument("--XAI_lambda", default = 0.2, type= float)
parser.add_argument("--name", default = "", type=str)

# try:  # terminal
#     parser = parser.parse_args()
# except:  # Jupyter notebook
parser = parser.parse_args(args=[])


# +

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

train_dataset = DataLoader(train_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=True, drop_last=True)

val_dataset = DataLoader(val_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)

# test_dataset = DataLoader(test_loader, batch_size=parser.batch_size, num_workers=parser.workers,
#                                 pin_memory=True, collate_fn=ArgoverseDataset.collate,
#                                 shuffle=False, drop_last=False)


model = WIMP(parser)
model.load_state_dict(torch.load("experiments/example_old/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) # 학습할 때에는 graph 모듈에서 p에 해당하는 network가 없었으므로

model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

conv_weight_origin = copy.deepcopy(model.decoder.xy_conv_filters[0].weight)
last_weight_origin = copy.deepcopy(model.decoder.value_generator.weight)

def get_metric(metric_dict, ade,fde,mr,loss, length):
    metric_dict["ade"] += (ade * length).cpu().item()
    metric_dict["fde"] += (fde * length).cpu().item()
    metric_dict["mr"] += (mr * length).cpu().item()
    metric_dict["loss"] += (loss * length).cpu().item()
    metric_dict["length"]+=length

Relu = nn.ReLU()

save_foler = "ResultsImg/"
save_XAI = save_foler + "/XAI/"
save_attention = save_foler + "/attention"

slicing = lambda a, idx: torch.cat((a[:idx], a[idx+1:]), axis=1)
slicing_2Dpadding = lambda a, idx: torch.cat((a[:idx], a[idx+1:], torch.zeros_like(a[0:1])), axis=0)
slicing_1Dpadding = lambda a, idx: F.pad(torch.cat((a[:idx], a[idx+1:]), axis=0), (0,1))

import time

now = time.localtime()
start_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
start_time = parser.name + str(parser.XAI_lambda) + start_time
save_folder = "results_XAI/" + start_time + "___" + str(parser.XAI_lambda).replace(".", "_") + "/"

os.mkdir(save_folder)
os.mkdir("ResultsImg/" + start_time)

def metric_to_dict(p, waypoint_preds, input_dict, target_dict, att_weights, i, weight):
    write_dict = {}
    write_dict['preds'] = p.tolist() 
    write_dict['waypoint_preds'] = [waypoint_preds[0][i].tolist(), waypoint_preds[1][i].tolist()]
    write_dict['rotation'] = input_dict['ifc_helpers']['rotation'][i].tolist()
    write_dict['translation'] = input_dict['ifc_helpers']['translation'][i].tolist()
    write_dict['csv_file'] = input_dict['ifc_helpers']['csv_file'][i]
    write_dict['city'] = str(input_dict['ifc_helpers']['city'][i])
    write_dict['agent_labels'] = target_dict['agent_labels'][i].tolist()
    write_dict['agent_features'] = input_dict['agent_features'][i].tolist()
    write_dict['social_features'] = input_dict['social_features'][i].tolist()
    write_dict['social_label_features'] = input_dict['social_label_features'][i].tolist()
    write_dict['att_weights'] = att_weights[i].tolist()
    write_dict["weight"] = weight.tolist()
    write_dict["mask"] = input_dict["num_agent_mask"][i].tolist()
    write_dict["gt"] = target_dict["agent_labels"][i].tolist()
    return write_dict

abs_min = lambda weight, k: torch.topk(abs(weight)[1:], k+1, largest = False).indices[k].item()
abs_max = lambda weight, k: torch.topk(abs(weight)[1:], k+1).indices[k].item()
simple_min = lambda weight, k: torch.topk(weight[1:], k+1, largest = False).indices[k].item()
simple_max = lambda weight, k: torch.topk(weight[1:], k+1).indices[k].item()


# +
def Test(dataset, f, f_idx=None):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False

    metrics = {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }
    
    for batch_idx, batch in enumerate(tqdm(dataset)):
        input_dict, target_dict = batch[0], batch[1]

        # get cuda
        input_dict["agent_features"] = input_dict["agent_features"].cuda()
        input_dict["social_features"] = input_dict["social_features"].cuda()
        input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
        input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
        input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["adjacency"] = input_dict["adjacency"].cuda()
        
        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
        input_dict["adjacency"] =f(input_dict["adjacency"])
        
#         num_agent_mask = input_dict['num_agent_mask'] 
#         input_dict["adjacency"] = input_dict["adjacency"] * num_agent_mask.unsqueeze(1) * num_agent_mask.unsqueeze(2)        
        input_dict['adjacency'][:,0,0] = input_dict['adjacency'][:,0,0].clamp(min = 1.0)

        with torch.no_grad():
            preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)

            loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)

            get_metric(metrics, ade, fde, mr, loss, len(input_dict["adjacency"]))

    write_json = {
        "metric": calc_mean(metrics)
    }
    print(write_json)


# -

relu = torch.nn.functional.relu


# +
# for batch_idx, batch in enumerate(tqdm(val_dataset)):

#     input_dict, target_dict = batch[0], batch[1]

#     # get cuda
#     input_dict["agent_features"] = input_dict["agent_features"].cuda()
#     input_dict["social_features"] = input_dict["social_features"].cuda()
#     input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
#     input_dict["adjacency"] = input_dict["adjacency"].cuda()
#     input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
#     input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
#     input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
#     input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
#     input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
#     input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()

#     target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
#     if batch_idx > 0:
#         break
# #     input_dict["adjacency"] =f(input_dict["adjacency"])
    
# -

model = model.cuda()

# +
# f1 = lambda w: softmax(normalize_max1(abs(w))*2) * len(w[0])
# f1 = lambda w: (normalize_max1((w+5000).clamp(min = 0.0))* 10000).clamp(max = 1.0)
# f1 = lambda w: (softmax(w) * len(w[0])) ** 0.001
# f2 = lambda w: (softmax(w) * len(w[0])) ** 0.005
# f3 = lambda w: (softmax(w) * len(w[0])) ** 0.01
# f4 = lambda w: (softmax(w) * len(w[0])) ** 0.05
# f5 = lambda w: (softmax(w) * len(w[0])) ** 0.06
# f6 = lambda w: (softmax(w) * len(w[0])) ** 0.07
# f7 = lambda w: (softmax(w) * len(w[0])) ** 0.08
# f8 = lambda w: (softmax(w) * len(w[0])) ** 0.09
# f9 = lambda w: (softmax(w) * len(w[0])) ** 0.10

# f7 = lambda w: (softmax(normalize_max1(abs(w))) * len(w[0])).clamp(min = 0.0, max=2.0)
# f2 = lambda w: softmax(normalize_max1(relu(w))) * len(w[0])
# f3 = lambda w: softmax(normalize_max1(abs(w))*2) * len(w[0])
# f4 = lambda w: softmax(normalize_max1(relu(w))*2) * len(w[0])
# f5 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
# f6 = lambda w: softmax(normalize_max1(relu(w))*0.5) * len(w[0])


error_data = []
f1 = lambda w: softmax(normalize_max1(abs(w))) * len(w[0])
f2 = lambda w: softmax(normalize_max1(relu(w) + 0.000001)) * len(w[0])
f3 = lambda w: softmax(normalize_max1(abs(w))*2) * len(w[0])
f4 = lambda w: softmax(normalize_max1(relu(w))*2 + 0.000001) * len(w[0])
f5 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
f6 = lambda w: softmax(normalize_max1(relu(w))*0.5 + 0.000001) * len(w[0])

f7 = lambda w: softmax(normalize_max1(w)) * len(w[0])
f8 = lambda w: softmax(normalize_max1(w)*2) * len(w[0])
f9 = lambda w: softmax(normalize_max1(w)*0.5) * len(w[0])

f10 = lambda w: softmax(abs(w)) * len(w[0])
f11 = lambda w: softmax(relu(w)) * len(w[0])
f12 = lambda w: softmax(abs(w)*2) * len(w[0])
f13 = lambda w: softmax(relu(w)*2) * len(w[0])
f14 = lambda w: softmax(abs(w)*0.5) * len(w[0])
f15 = lambda w: softmax(relu(w)*0.5) * len(w[0])


f16 = lambda w: relu(normalize_max1(w)) * len(w[0])
f17 = lambda w: abs(normalize_max1(w)) * len(w[0])
f18 = lambda w: relu(normalize_max1(w)*0.5) * len(w[0])
f19 = lambda w: abs(normalize_max1(w)*0.5) * len(w[0])
f20 = lambda w: relu(normalize_max1(w)*2) * len(w[0])
f21 = lambda w: abs(normalize_max1(w)*2) * len(w[0])

f22 = lambda w: normalize_max1(w) * len(w[0])
f23 = lambda w: normalize_max1(w) * len(w[0])
f24 = lambda w: normalize_max1(w)*0.5 * len(w[0])
f25 = lambda w: normalize_max1(w)*0.5 * len(w[0])
f26 = lambda w: normalize_max1(w)*2 * len(w[0])
f27 = lambda w: normalize_max1(w)*2 * len(w[0])


to_gaussian = lambda arr, mean = 0, std = 1: ((arr - torch.mean(arr))/ torch.std(arr) ) * std + mean



# for f_idx, f in enumerate([ f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27]):
for f_idx, f in enumerate([0]):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False

    metrics = {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }
    
    for batch_idx, batch in enumerate(tqdm(val_dataset)):

        input_dict, target_dict = batch[0], batch[1]

        # get cuda
        input_dict["agent_features"] = input_dict["agent_features"].cuda()
        input_dict["social_features"] = input_dict["social_features"].cuda()
        input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
        input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
        input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["adjacency"] = input_dict["adjacency"].cuda()
        
        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
        for i in range(len(input_dict["adjacency"])):
            num_agent = int(torch.sum(input_dict['num_agent_mask'][0]))
            input_dict["adjacency"][0,0, :num_agent] = to_gaussian(input_dict["adjacency"][0,0, :num_agent], mean = 1, std = 0.05)
            
        
        num_agent_mask = input_dict['num_agent_mask'] 
        input_dict["adjacency"] = input_dict["adjacency"] * num_agent_mask.unsqueeze(1) * num_agent_mask.unsqueeze(2)        
        
        with torch.no_grad():
            preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)

            loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)


            get_metric(metrics, ade, fde, mr, loss, len(input_dict["adjacency"]))

    if parser.remove_high_related_score:
        for i in range(len(DA_model_metric)):
            DA_model_metric[i] = calc_mean(DA_model_metric[i])
    else:
        DA_model_metric = None
        
    metrics = calc_mean(metrics)

    write_json = {
        "metric": metrics,
        "XAI_lambda": parser.XAI_lambda
    }
    print(f_idx)
    print(write_json)
# -

to_gaussian(input_dict["adjacency"][0,0][:6], mean = 1, std = 0.05)

num_agent = int(torch.sum(input_dict['num_agent_mask'][0]))
input_dict["adjacency"][0,0, :num_agent]

input_dict['num_agent_mask'][0]

input_dict["adjacency"].shape


# +
f1 = lambda w: (softmax(w) * len(w[0])) ** 0.01
f2 = lambda w: (softmax(w) * len(w[0])) ** 0.05
f3 = lambda w: (softmax(w) * len(w[0])) ** 0.1
f4 = lambda w: (softmax(w) * len(w[0])) ** 0.2
f5 = lambda w: (softmax(w) * len(w[0])) ** 0.3
f6 = lambda w: (softmax(w) * len(w[0])) ** 0.4
f7 = lambda w: (softmax(w) * len(w[0])) ** 0.5
f8 = lambda w: (softmax(w) * len(w[0])) ** 0.6
f9 = lambda w: (softmax(w) * len(w[0])) ** 0.7
f10 = lambda w: (softmax(w) * len(w[0])) ** 0.8
f11 = lambda w: (softmax(w) * len(w[0])) ** 0.9
f12 = lambda w: (softmax(w) * len(w[0])) ** 1.0
f13 = lambda w: (softmax(w) * len(w[0])) ** 1.1
f14 = lambda w: (softmax(w) * len(w[0])) ** 1.2
f15 = lambda w: (softmax(w) * len(w[0])) ** 1.3
f16 = lambda w: (softmax(w) * len(w[0])) ** 1.4
f17 = lambda w: (softmax(w) * len(w[0])) ** 1.5
f18= lambda w: (softmax(w) * len(w[0])) ** 1.6
f19 = lambda w: (softmax(w) * len(w[0])) ** 1.7
f20 = lambda w: (softmax(w) * len(w[0])) ** 1.8
f21 = lambda w: (softmax(w) * len(w[0])) ** 1.9
f22 = lambda w: (softmax(w) * len(w[0])) ** 2.0

f23 = lambda w: (softmax(normalize_max1(abs(w))) * len(w[0])).clamp(min = 0.0, max=2.0)
f24 = lambda w: (softmax(normalize_max1(relu(w) + 0.00001)) * len(w[0])).clamp(min = 0.0, max=2.0)
f25 = lambda w: (softmax(normalize_max1(abs(w))*2) * len(w[0])).clamp(min = 0.0, max=2.0)
f26 = lambda w: (softmax(normalize_max1(relu(w) + 0.00001)*2) * len(w[0])).clamp(min = 0.0, max=2.0)
f27 = lambda w: (softmax(normalize_max1(abs(w))*0.5) * len(w[0])).clamp(min = 0.0, max=2.0)
f28 = lambda w: (softmax(normalize_max1(relu(w) + 0.00001)*0.5) * len(w[0])).clamp(min = 0.0, max=2.0)





for f_idx, f in enumerate([f23, f24, f25, f26, f27, f28]):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False

    metrics = {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }
    
    for batch_idx, batch in enumerate(tqdm(val_dataset)):

        input_dict, target_dict = batch[0], batch[1]

        # get cuda
        input_dict["agent_features"] = input_dict["agent_features"].cuda()
        input_dict["social_features"] = input_dict["social_features"].cuda()
        input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
        input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
        input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["adjacency"] = input_dict["adjacency"].cuda()
        
        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
        input_dict["adjacency"] =f(input_dict["adjacency"])
        
        num_agent_mask = input_dict['num_agent_mask'] 
        input_dict["adjacency"] = input_dict["adjacency"] * num_agent_mask.unsqueeze(1) * num_agent_mask.unsqueeze(2)        
        

        with torch.no_grad():
            preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)

            loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)

            get_metric(metrics, ade, fde, mr, loss, len(input_dict["adjacency"]))

    if parser.remove_high_related_score:
        for i in range(len(DA_model_metric)):
            DA_model_metric[i] = calc_mean(DA_model_metric[i])
    else:
        DA_model_metric = None
        
    metrics = calc_mean(metrics)

    write_json = {
        "metric": metrics,
        "XAI_lambda": parser.XAI_lambda
    }
    print(f_idx)
    print(write_json)
    
# -

a = torch.tensor([[[0,0,0]]]).float() + 1
f24(a)

model.load_state_dict(torch.load("experiments/example_old/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) # 학습할 때에는 graph 모듈에서 p에 해당하는 network가 없었으므로


for batch_idx, batch in enumerate(tqdm(val_dataset)):
    input_dict, target_dict = batch[0], batch[1]
    print(input_dict['adjacency'][0][0])
    if batch_idx > 5:
        break

# +
f1 = lambda w: (softmax(w) * len(w[0]))
f1 = lambda w: torch.exp(w.clamp(max = 2))
f2 = lambda w: f1(w) *(f1(w) > 0.998)
# f2 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])

f3 = lambda w: torch.exp(w.clamp(max = 2))

f2(input_dict['adjacency'])[15][0], input_dict['adjacency'][:,0,0].shape
# -

input_dict['adjacency'][10][0]

f1(input_dict['adjacency'][10][0])

# +
# f1 = lambda w: (softmax(normalize_max1(abs(w))) * len(w[0])).clamp(min = 0.0, max=2.0)
# f2 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])


# f1 = lambda w: torch.exp(w.clamp(max = 2)) ** 0.5
# f2 = lambda w: torch.exp(w.clamp(max = 2)) ** 1
# f3 = lambda w: torch.exp(w.clamp(max = 2)) ** 2

# f12 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
# f12 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
# f2 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
# f1 = lambda w: -(torch.exp(w.clamp(max = 2)) ** 0.5)
# f2 = lambda w: torch.exp(w.clamp(max = 2)) ** 2
# f3 = lambda w: torch.ones_like(w)

for f_idx, f in enumerate([f1, f2, f3]):#, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27, f28]):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False

    metrics = {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }
    
    for batch_idx, batch in enumerate(tqdm(val_dataset)):

        input_dict, target_dict = batch[0], batch[1]

        # get cuda
        input_dict["agent_features"] = input_dict["agent_features"].cuda()
        input_dict["social_features"] = input_dict["social_features"].cuda()
        input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
        input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
        input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
        input_dict["adjacency"] = input_dict["adjacency"].cuda()
        
        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
        input_dict["adjacency"] =f(input_dict["adjacency"])
        
        num_agent_mask = input_dict['num_agent_mask'] 
#         input_dict["adjacency"] = input_dict["adjacency"] * num_agent_mask.unsqueeze(1) * num_agent_mask.unsqueeze(2)        
#         input_dict['adjacency'][:,0,0] = input_dict['adjacency'][:,0,0].clamp(min = 1.0)

        with torch.no_grad():
            preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)

            loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)

            get_metric(metrics, ade, fde, mr, loss, len(input_dict["adjacency"]))

    if parser.remove_high_related_score:
        for i in range(len(DA_model_metric)):
            DA_model_metric[i] = calc_mean(DA_model_metric[i])
    else:
        DA_model_metric = None
        
    metrics = calc_mean(metrics)

    write_json = {
        "metric": metrics,
        "XAI_lambda": parser.XAI_lambda
    }
    print(f_idx)
    print(write_json)
    
# -

{'metric': {'fde': 1.146207567647154, 'ade': 0.7541295413213265, 'loss': 6.052047431831375, 'mr': 0.11783035856342818}, 'XAI_lambda': 0.2}


input_dict["adjacency"]

calc_mean(metrics)

od['adjacency_lrp'][0]

for e in error_data:
    e, e2 = e
    with open(e['ifc_helpers']['file_path'][0].replace('LRP_adjacency', 'argoverse_with_LRP'), 'rb') as ff:
        od = pickle.load(ff)
#     if od['metric']['fde'] > 2.5:
    print(e2, float(torch.std(e['adjacency'][0,0])),
    float(torch.max(e['adjacency'][0,0])),
    float(torch.min(e['adjacency'][0,0])))

fde, od['metric']['fde']

torch.std(e['adjacency'][0,0])

error_data[0][0]['adjacency']

error_data[0][1]

od['metric']['fde']

for e in error_data:
    e, e2 = e
    with open(e['ifc_helpers']['file_path'][0].replace('LRP_adjacency', 'argoverse_with_LRP'), 'rb') as ff:
        od = pickle.load(ff)
    if od['metric']['fde'] > 2.5:
        print(e2, float(torch.std(e['adjacency'][0,0])),
        float(torch.max(e['adjacency'][0,0])),
        float(torch.min(e['adjacency'][0,0])))

with open(e['ifc_helpers']['file_path'][0].replace('LRP_adjacency', 'argoverse_with_LRP'), 'rb') as ff:
    od = pickle.load(ff)

od['metric']['fde']

# +
f1 = lambda w: softmax(normalize_max1(abs(w))) * len(w[0])
f2 = lambda w: softmax(normalize_max1(relu(w))) * len(w[0])
f3 = lambda w: softmax(normalize_max1(abs(w))*2) * len(w[0])
f4 = lambda w: softmax(normalize_max1(relu(w))*2) * len(w[0])
f5 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
f6 = lambda w: softmax(normalize_max1(relu(w))*0.5) * len(w[0])

input_dict['adjacency'] = f1(input_dict['adjacency'])
# -

fde

preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)

loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)


fde

# +
f1 = lambda w: softmax(normalize_max1(abs(w))) * len(w[0])
f2 = lambda w: softmax(normalize_max1(relu(w))) * len(w[0])
f3 = lambda w: softmax(normalize_max1(abs(w))*2) * len(w[0])
f4 = lambda w: softmax(normalize_max1(relu(w))*2) * len(w[0])
f5 = lambda w: softmax(normalize_max1(abs(w))*0.5) * len(w[0])
f6 = lambda w: softmax(normalize_max1(relu(w))*0.5) * len(w[0])

f7 = lambda w: softmax(normalize_max1(w)) * len(w[0])
f8 = lambda w: softmax(normalize_max1(w)*2) * len(w[0])
f9 = lambda w: softmax(normalize_max1(w)*0.5) * len(w[0])

f10 = lambda w: softmax(abs(w)) * len(w[0])
f11 = lambda w: softmax(relu(w)) * len(w[0])
f12 = lambda w: softmax(abs(w)*2) * len(w[0])
f13 = lambda w: softmax(relu(w)*2) * len(w[0])
f14 = lambda w: softmax(abs(w)*0.5) * len(w[0])
f15 = lambda w: softmax(relu(w)*0.5) * len(w[0])


f16 = lambda w: relu(normalize_max1(w)) * len(w[0])
f17 = lambda w: abs(normalize_max1(w)) * len(w[0])
f18 = lambda w: relu(normalize_max1(w)*0.5) * len(w[0])
f19 = lambda w: abs(normalize_max1(w)*0.5) * len(w[0])
f20 = lambda w: relu(normalize_max1(w)*2) * len(w[0])
f21 = lambda w: abs(normalize_max1(w)*2) * len(w[0])

f22 = lambda w: normalize_max1(w) * len(w[0])
f23 = lambda w: normalize_max1(w) * len(w[0])
f24 = lambda w: normalize_max1(w)*0.5 * len(w[0])
f25 = lambda w: normalize_max1(w)*0.5 * len(w[0])
f26 = lambda w: normalize_max1(w)*2 * len(w[0])
f27 = lambda w: normalize_max1(w)*2 * len(w[0])


for f_idx, f in enumerate([f1, f2, f3, f4, f5, f6, f7, f8, f9, f10, f11, f12, f13, f14, f15, f16, f17, f18, f19, f20, f21, f22, f23, f24, f25, f26, f27]):
    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False

    metrics = {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }
    
    for batch_idx, batch in enumerate(tqdm(val_dataset)):

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
        input_dict["adjacency"] =f(input_dict["adjacency"])
                
        with torch.no_grad():
            preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)
            loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
#             loss.backward()
#             print(adjacency.grad)
#             assert adjacency.grad != None, "adjacency_grad is None"
#             assert gan_features.grad != None, "gan_features_grad is None"

#             optimizer.step()

            get_metric(metrics, ade, fde, mr, loss, len(input_dict["adjacency"]))

            
    if parser.remove_high_related_score:
        for i in range(len(DA_model_metric)):
            DA_model_metric[i] = calc_mean(DA_model_metric[i])
    else:
        DA_model_metric = None
        
    metrics = calc_mean(metrics)

    write_json = {
        "metric": metrics,
        "XAI_lambda": parser.XAI_lambda
    }
    print(f_idx)
    print(write_json)

# -

for batch_idx, batch in enumerate(tqdm(val_dataset)):

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
    break
#     input_dict["adjacency"] =f(input_dict["adjacency"])

