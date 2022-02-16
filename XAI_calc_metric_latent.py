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
from src.models.WIMP import WIMP
from argoverse.map_representation.map_api import ArgoverseMap

import XAI_utils

import torch.backends.cudnn as cudnn
import random
from argparse import ArgumentParser
parser = ArgumentParser()


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




slicing = lambda a, idx: torch.cat((a[:idx], a[idx+1:]), axis=1)

# if 10 명 에이전트 중 1개를 지웠다면 지운놈을 0으로 패딩 -> slicing padding 1D padding -> padding 바꾸기 

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


for name_idx, function in enumerate([abs_min, abs_max, simple_min, simple_max]):
    torch.manual_seed(0) 
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    
    
    original_model_metric = {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }

    DA_model_metric = [
        {
            "ade": 0.0,
            "fde": 0.0,
            "mr": 0.0,
            "loss": 0.0,
            "length": 0,
        },
        {
            "ade": 0.0,
            "fde": 0.0,
            "mr": 0.0,
            "loss": 0.0,
            "length": 0,
        },
        {
            "ade": 0.0,
            "fde": 0.0,
            "mr": 0.0,
            "loss": 0.0,
            "length": 0,
        },
    ]  # 하나씩 지우면서 metric을 잴것임
    
    adjacency_exp = {
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

        preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)
        
        adjacency.retain_grad()
        gan_features.retain_grad()
        graph_output.retain_grad()
        input_dict["social_features"].retain_grad()
        
        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        loss.backward()

        input_dict["social_features"].grad
        print(input_dict["social_features"].grad)

        # grad 없을수도 있수도 있어서 
        assert adjacency.grad != None, "adjacency_grad is None"
        assert gan_features.grad != None, "gan_features_grad is None"

        # optimizer.step()
        adjacency_grad = adjacency * adjacency.grad
        feature_grad = torch.sum(gan_features.grad * gan_features, axis=3).squeeze(-1)
        
        get_metric(adjacency_exp, ade, fde, mr, loss,len(adjacency_grad)) 
        
        if parser.remove_high_related_score:
    
            with torch.no_grad():
                
                for i in range(parser.maximum_delete_num):  # 하나씩 지우자
                    
                    for idx in range(len(input_dict["agent_features"])): # batch 안의 iteration별로
                        
                        num_vehicles =  int(torch.sum(input_dict["num_agent_mask"][idx]))

                        if parser.use_hidden_feature:
                            weight = feature_grad[idx]
                        else:
                            weight = adjacency_grad[idx][0]
                        
                        if num_vehicles > 1: # social agent가 1이상일때, 즉 첫번째 social agent의 data가 존재하는 경우
                            arg = function(weight[:num_vehicles], 0)
                            weight = torch.cat((weight[: arg + 1], weight[arg + 2 :]))

                            input_dict["num_agent_mask"][idx] = slicing_1Dpadding(input_dict["num_agent_mask"][idx], arg + 1)
                            adjacency_grad[idx][0] = slicing_1Dpadding(adjacency_grad[idx][0], arg+1) # adjacency 지움
                            feature_grad[idx] = slicing_1Dpadding(feature_grad[idx], arg+1) # feature_grad지움

                            input_dict["social_features"][idx] = slicing_2Dpadding(input_dict["social_features"][idx], arg) #  agent 지움
                            input_dict["ifc_helpers"]["social_oracle_centerline"][idx] = slicing_2Dpadding(input_dict["ifc_helpers"]["social_oracle_centerline"][idx], arg)
                            input_dict["ifc_helpers"]["social_oracle_centerline_lengths"][idx] = slicing_2Dpadding(input_dict["ifc_helpers"]["social_oracle_centerline_lengths"][idx],arg,)
    
                    preds_,waypoint_preds_,_,_,_,_,_= model(**input_dict)

                    loss, (ade, fde, mr) = model.eval_preds(preds_, target_dict, waypoint_preds_)
                    get_metric(DA_model_metric[i], ade, fde, mr, loss, len(adjacency_grad))

                    if parser.save_json:
                        write_dict = copy.deepcopy([metric_to_dict(preds_[j], waypoint_preds_, input_dict, target_dict, attention,j, adjacency_grad[j][0]) for j in range(len(preds))])
                        write_json_delete[name_idx][i] += write_dict
                

        adjacency.detach()
        gan_features.detach()

        break 

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
    
    if parser.remove_high_related_score:
        for i in range(len(DA_model_metric)):
            DA_model_metric[i] = calc_mean(DA_model_metric[i])
    else:
        DA_model_metric = None
        
    adjacency_exp = calc_mean(adjacency_exp)

    write_json = {
        "DA_model_metric": DA_model_metric,
        "NON_DA_model_metric": adjacency_exp,
        "XAI_lambda": parser.XAI_lambda
    }

    print("SAVE : ", save_folder + names[name_idx] + ".json")
    with open(save_folder + names[name_idx]  + ".json", "w") as json_data:
        json.dump(write_json, json_data, indent=4)

    print(write_json)
    

