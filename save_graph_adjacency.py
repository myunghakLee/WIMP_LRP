# -*- coding: utf-8 -*-
# +
from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import copy
import json
import os

from src.data.argoverse_datamodule import ArgoverseDataModule
from src.data.argoverse_dataset import ArgoverseDataset
from src.data.dummy_datamodule import DummyDataModule
from src.models.WIMP import WIMP

os.environ["CUDA_VISIBLE_DEVICES"]= "2"

# + endofcell="--"
args = {"IFC":True, "add_centerline":False, "attention_heads":4, "batch_norm":False, "batch_size":64, "check_val_every_n_epoch":3, 
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
          "save_json": False, "make_submit_file" : False, "use_hidden_feature" : True, "is_LRP": True, "adjacency_exp" : True}


from argparse import ArgumentParser
parser = ArgumentParser()

for k in args:
    parser.add_argument(str("--" + k), default = args[k], type= type(args[k]))
parser.add_argument("--XAI_lambda", default = 0.0, type= float)
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
model.load_state_dict(torch.load("experiments/example_old/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) # ????????? ????????? graph ???????????? p??? ???????????? network??? ???????????????
# model =nn.parallel.DataParallel(model)

model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

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



#  optimizer??? ???????????? ???????????????.
#  ????????? ?????? model weight??? gradient??? ?????? ??????????????? ????????? ????????? ?????? adjacency matrix??? graident????????? ?????? ?????? ????????? ???????????????.
import os
import math
import copy
Relu = nn.ReLU()

save_foler = "ResultsImg/"

# # +
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

num_socials = []
names = ["abs_min", "abs_max", "simple_min", "simple_max"]
write_json_original = []

write_json_delete = [[[],[],[]] , [[],[],[]], [[],[],[]], [[],[],[]]]  #[names][delete_num][data_iter]
# --

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
import pickle

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
torch.backends.cudnn.benchmark = False

for dataset in [val_dataset, train_dataset]:
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

        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()

        preds, waypoint_preds, all_dist_params, attention, adjacency, gan_features, graph_output = model(**input_dict)
        adjacency.retain_grad()
        gan_features.retain_grad()


        loss, _ = model.eval_preds(preds, target_dict, waypoint_preds)
        loss.backward()
    #         print(adjacency)
    #         print(adjacency.grad)
    #     print(adjacency.grad * adjacency)
    #     print(softmax(adjacency.grad * adjacency) * len(adjacency[0]))################ ????????? ????????? ?????? ?????? ?????????(?????? ????????? ?????? ???????)
    #     break

        assert adjacency.grad != None, "adjacency_grad is None"
        assert gan_features.grad != None, "gan_features_grad is None"

        optimizer.zero_grad()
        adjacency_grad = adjacency * adjacency.grad
        feature_grad = torch.sum(gan_features.grad * gan_features, axis=3).squeeze(-1)

        for ii in range(len(input_dict['ifc_helpers']['file_path'])):
            load_path = input_dict['ifc_helpers']['file_path'][ii]
            save_path = input_dict['ifc_helpers']['file_path'][ii].replace("argoverse_processed_simple", "LRP_adjacency3")

            with open(load_path, 'rb') as f:
                d = pickle.load(f)
            length = int(sum(input_dict["num_agent_mask"][ii]))
            pad_num = len(d['SOCIAL'])+1 - length
#             zero_pasdding =  nn.ZeroPad2d((0,pad_num,0,pad_num))
            zero_pasdding =  nn.ZeroPad2d((0,pad_num,0,pad_num+length-1))

            d['ADJACENCY'] = zero_pasdding(feature_grad[ii][:length].unsqueeze(0)).cpu().detach().tolist()
            with open(save_path, "wb") as f:
                pickle.dump(d, f)

# -


save_path

feature_grad.shape[ii]

feature_grad[ii][:length].unsqueeze(-1).shape

zero_pasdding(feature_grad[ii][:length].unsqueeze(0))

with open(save_path, 'rb') as f:
    d = pickle.load(f)

zero_pasdding =  nn.ZeroPad2d((0,2,0,2))
adjacency_grad[ii].shape
zero_pasdding(adjacency_grad[ii]).shape

input_dict["social_features"][0][11]

int(sum(input_dict["num_agent_mask"][0]))

# +

with open(input_dict["ifc_helpers"]["file_path"][0].replace("argoverse_processed_simple", "argoverse_with_LRP"), "rb") as f:
    d = pickle.load(f)
len(d['SOCIAL']), len(d['TE'])#, len(d['ADJACENCY'])
# -

len(d['SOCIAL'])+1, len(d['TE']), len(d['ADJACENCY'])

d.keys()

d["SOCIAL"][0].keys()

input_dict["social_label_features"].shape
