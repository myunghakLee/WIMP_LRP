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

args = {"IFC":True, "add_centerline":False, "attention_heads":4, "batch_norm":False, "batch_size":25, "check_val_every_n_epoch":3, 
          "dataroot":'./data/argoverse_processed_simple', "dataset":'argoverse', "distributed_backend":'ddp', "dropout":0.0, 
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

try:  # terminal
    parser = parser.parse_args()
except:  # Jupyter notebook
    parser = parser.parse_args(args=[])


train_loader = ArgoverseDataset(parser.dataroot, mode='train', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

val_loader = ArgoverseDataset(parser.dataroot, mode='val', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

test_loader = ArgoverseDataset(parser.dataroot, mode='test', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

train_dataset = DataLoader(train_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=True, drop_last=True)

val_dataset = DataLoader(val_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)

test_dataset = DataLoader(test_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)


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
softmax = torch.nn.Softmax(dim=2)

def calc_mean(metric):
    metric["fde"] /= metric["length"]
    metric["ade"] /= metric["length"]
    metric["mr"] /= metric["length"]
    metric["loss"] /= metric["length"]
    return {
        "fde": metric["fde"],
        "ade": metric["ade"],
        "loss": metric["loss"],
        "mr": metric["mr"]
    }



# +
for batch_idx, batch in enumerate(tqdm(val_dataset)):
    input_dict, target_dict = batch[0], batch[1]
    input_dict["adjacency"] = softmax(input_dict["adjacency"]) * len(input_dict['adjacency'][0])
    print(input_dict['adjacency'].shape)
    
#     print(softmax(input_dict['adjacency'][:, 0, :, :]) * len(input_dict['adjacency'][0][0]))
    break

# +

for epoch_id in range(10):
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
        
        input_dict["adjacency"] = softmax(input_dict["adjacency"]) * len(input_dict['adjacency'][0])
#         with torch.no_grad():
        preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)
        adjacency.retain_grad()
#         optimizer.zero_grad()
        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        loss.backward(retain_graph=True)

#         assert adjacency.grad != None, "adjacency_grad is None"
#         assert gan_features.grad != None, "gan_features_grad is None"

#         optimizer.step()

        get_metric(metrics, ade, fde, mr, loss, len(input_dict["adjacency"]))

            
    if parser.remove_high_related_score:
        for i in range(len(DA_model_metric)):
            DA_model_metric[i] = calc_mean(DA_model_metric[i])
    else:
        DA_model_metric = None
        
    metrics = calc_mean(metrics)

    write_json = {
        "adjacency_exp": metrics,
        "XAI_lambda": parser.XAI_lambda
    }
    print(write_json)
    break


# +
def func(a,b,c,d):
    with torch.backends.cudnn.flags(enabled=False):
        preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(a, b, c, d, ifc_helpers = input_dict["ifc_helpers"])
        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        loss = loss.cpu()
        print(loss)
        return loss
    
torch.autograd.functional.hessian(func,tuple([input_dict["agent_features"], input_dict["social_features"], input_dict["adjacency"],
                                                                                   input_dict["num_agent_mask"]]))
# -



model.encoder.xy_conv_filters[0].weight.grad

input_dict["adjacency"].shape

input_dict["adjacency"]

# +
metric_exp = {
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
#         input_dict["adjacency"] = input_dict["adjacency"].cuda()
    input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
    input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
    input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()
    input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"].cuda()
    input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict["ifc_helpers"]["social_oracle_centerline"].cuda()
    input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict["ifc_helpers"]["social_oracle_centerline_lengths"].cuda()
    input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict["ifc_helpers"]["agent_oracle_centerline"].cuda()

    target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
    with torch.no_grad():
        preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)


        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        get_metric(metric_exp, ade, fde, mr, loss,len(adjacency_grad))

metric_exp = calc_mean(metric_exp)
print(metric_exp)
# -

with open("train_with_XAI.json", "w") as j:
    json.dump(metric_exp, j)

metrics

adjacency_exp

# softmax말고 0~1사이로
for batch_idx, batch in enumerate(tqdm(train_dataset)):
#     print(batch[0]['adjacency'])
    print(softmax(batch[0]['adjacency']) * len(batch[0]['adjacency'][0][0]))
#     print(softmax(batch[0]['adjacency']))
    break

len(input_dict["adjacency"])

