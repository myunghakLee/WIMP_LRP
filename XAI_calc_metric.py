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



# +
import XAI_utils

import torch.backends.cudnn as cudnn
import random

# -

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
          }


# +
from argparse import ArgumentParser
parser = ArgumentParser()


for k in args:
    parser.add_argument(str("--" + k), default = args[k], type= type(args[k]))
parser.add_argument("--XAI_lambda", default = 0.2, type= float)

try:  # terminal
    parser = parser.parse_args()
except:  # Jupyter notebook
    parser = parser.parse_args(args=[])


# -

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

trest_dataset = DataLoader(test_loader, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)
# dm = ArgoverseDataset(args)

# +
model = WIMP(parser)
model.load_state_dict(torch.load("experiments/example/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) # 학습할 때에는 graph 모듈에서 p에 해당하는 network가 없었으므로
# model =nn.parallel.DataParallel(model)
model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.00, momentum=0.9)

# conv_weight_origin = copy.deepcopy(model.module.decoder.xy_conv_filters[0].weight)
# last_weight_origin = copy.deepcopy(model.module.decoder.value_generator.weight)
conv_weight_origin = copy.deepcopy(model.decoder.xy_conv_filters[0].weight)
last_weight_origin = copy.deepcopy(model.decoder.value_generator.weight)

# -

def get_metric(metric_dict, ade,fde,mr,loss, length):
    metric_dict["ade"] += (ade * length).cpu().item()
    metric_dict["fde"] += (fde * length).cpu().item()
    metric_dict["mr"] += (mr * length).cpu().item()
    metric_dict["loss"] += (loss * length).cpu().item()
    metric_dict["length"]+=length


#  optimizer는 선언하지 않았습니다.
#  따라서 아마 model weight에 gradient는 계속 쌓이겠지만 저희가 중요한 것은 adjacency matrix의 graident이므로 상관 없을 것으로 예측됩니다.
import os
import math
import copy
Relu = nn.ReLU()

save_foler = "ResultsImg/"

# +
save_XAI = save_foler + "/XAI/"
save_attention = save_foler + "/attention"

# pad2D = nn.ZeroPad2d((0,1,0,1))
# pad1D = F.pad(aa, (0,1))

slicing = lambda a, idx: torch.cat((a[:idx], a[idx+1:]), axis=1)
slicing_2Dpadding = lambda a, idx: torch.cat((a[:idx], a[idx+1:], torch.zeros_like(a[0:1])), axis=0)
slicing_1Dpadding = lambda a, idx: F.pad(torch.cat((a[:idx], a[idx+1:]), axis=0), (0,1))



# +
import time

now = time.localtime()
 
start_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)

save_folder = "results_XAI/" + start_time + "___" + str(parser.XAI_lambda).replace(".", "_") + "/"

os.mkdir(save_folder)


# +
abs_min = lambda weight, k: torch.topk(abs(weight)[1:], k+1, largest = False).indices[k].item()
abs_max = lambda weight, k: torch.topk(abs(weight)[1:], k+1).indices[k].item()
simple_min = lambda weight, k: torch.topk(weight[1:], k+1, largest = False).indices[k].item()
simple_max = lambda weight, k: torch.topk(weight[1:], k+1).indices[k].item()


names = ["abs_min", "abs_max", "simple_min", "simple_max"]
for name_idx, function in enumerate([abs_min, abs_max, simple_min, simple_max]):

    torch.manual_seed(0)
    torch.cuda.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(0)
    torch.backends.cudnn.benchmark = False
    
    optimizer.zero_grad()
    
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

    for batch_idx, batch in enumerate(tqdm(val_dataset)):
        input_dict, target_dict = batch[0], batch[1]

        # get cuda
#         optimizer.zero_grad()
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
        #         if batch_idx == 0:
        #             print(input_dict['social_features'][0,:,:2,:2])

        preds, waypoint_preds, all_dist_params, attention, adjacency = model(**input_dict)
            
        
        adjacency.retain_grad()

        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        loss.backward()
        get_metric(original_model_metric, ade, fde, mr, loss,len(adjacency.grad)) #####################수정 필요

#         #  sanity check
#         conv_weight = model.decoder.xy_conv_filters[0].weight
#         last_weight = model.decoder.value_generator.weight
#         assert torch.all(conv_weight == conv_weight_origin) and torch.all(last_weight == last_weight_origin), 
#             ("Model is Changed",conv_weight,conv_weight_origin,last_weight,last_weight_origin,)

        if parser.draw_image:
            for idx in range(args["batch_size"]):
                weight = adjacency.grad[idx][0].cpu().numpy()
                att = attention[idx].cpu().numpy()
                agent_features = input_dict["agent_features"][idx].cpu().numpy()
                social_features = input_dict["social_features"][idx].cpu().numpy()
                # target = target_dict['agent_labels'][idx].cpu().numpy()
                preds = batch_preds[idx][:, :, :, :2][0].cpu().detach().numpy()
                city_name = input_dict["ifc_helpers"]["city"][idx]
                rotation = input_dict["ifc_helpers"]["rotation"][idx].numpy()
                translation = input_dict["ifc_helpers"]["translation"][idx].numpy()
                XAI_utils.draw_attention(agent_features,social_features,preds,city_name,rotation,translation,weight=copy.deepcopy(att),draw_future=True,
                    save_fig=True,save_name=save_attention+ "/"+ str(batch_idx)+ "_"+ str(idx)+ ".png",)

                XAI_utils.draw(agent_features,social_features, preds, city_name, rotation, translation, weight=copy.deepcopy(weight), draw_future=True,
                    save_fig=True, save_name=save_XAI + "/" + str(batch_idx) + "_" + str(idx) + ".png")

        if parser.remove_high_related_score:  # 만약 XAI 실험을 할 것이면
            with torch.no_grad():
                for i in range(parser.maximum_delete_num):  # 하나씩 지우자
                    for idx in range(len(adjacency.grad)): # batch 안의 iteration별로
                        weight = adjacency.grad[idx][0]
                        if len(input_dict["social_features"][0] > 1):
                            # slicing
                            arg = function(weight, i)
                            weight = torch.cat((weight[: arg + 1], weight[arg + 2 :]))
                            input_dict["social_features"][idx] = slicing_2Dpadding(input_dict["social_features"][idx], arg)
                            input_dict["num_agent_mask"][idx] = slicing_1Dpadding(input_dict["num_agent_mask"][idx], arg + 1)
                            input_dict["ifc_helpers"]["social_oracle_centerline"][idx] = slicing_2Dpadding(input_dict["ifc_helpers"]["social_oracle_centerline"][idx], arg)
                            input_dict["ifc_helpers"]["social_oracle_centerline_lengths"][idx] = slicing_2Dpadding(input_dict["ifc_helpers"]["social_oracle_centerline_lengths"][idx],arg,)
                        else:
                            break

                    preds,waypoint_preds,all_dist_params,att_weights,_ = model(input_dict["agent_features"],
                        input_dict["social_features"],None,input_dict["num_agent_mask"],
                        ifc_helpers={"social_oracle_centerline": input_dict["ifc_helpers"]["social_oracle_centerline"],
                                        "social_oracle_centerline_lengths": input_dict["ifc_helpers"]["social_oracle_centerline_lengths"],
                                        "agent_oracle_centerline": input_dict["ifc_helpers"]["agent_oracle_centerline"],
                                        "agent_oracle_centerline_lengths": input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"],
                        })
                    loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
                    get_metric(DA_model_metric[i], ade, fde, mr, loss, len(adjacency.grad))

        torch.cuda.empty_cache()

        
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

    for i in range(len(DA_model_metric)):
        DA_model_metric[i] = calc_mean(DA_model_metric[i])
    original_model_metric = calc_mean(original_model_metric)

    write_json = {
        "DA_model_metric": DA_model_metric,
        "original_model_metric": original_model_metric,
        "XAI_lambda": parser.XAI_lambda
    }

    print("SAVE : ", save_folder + names[name_idx] + ".json")
    with open(save_folder + names[name_idx]  + ".json", "w") as json_data:
        json.dump(write_json, json_data, indent=4)

    print(write_json)

# +
import os

os.listdir("results_XAI")
