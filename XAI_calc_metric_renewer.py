# -*- coding: utf-8 -*-
# +
from argparse import ArgumentParser
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import random
import math
import time
import copy
import json
import os

from torch.utils.data import DataLoader, Dataset
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch

from src.data.argoverse_datamodule import ArgoverseDataModule
from src.data.argoverse_dataset import ArgoverseDataset
from src.data.dummy_datamodule import DummyDataModule

from src.models.WIMP import WIMP
import XAI_utils_renewer
# -

os.environ["CUDA_VISIBLE_DEVICES"]= "2"


parser = ArgumentParser()

# +
args = XAI_utils_renewer.get_args(batch_size=24, dataroot='./data/argoverse_processed_simple',  dropout=0.0, 
                                  experiment_name='example', draw_image = False, remove_high_related_score = True, 
                                  maximum_delete_num = 3, save_json= True, make_submit_file = False, is_LRP= True)

for k in args:
    parser.add_argument(str("--" + k), default = args[k], type= type(args[k]))
    
parser.add_argument("--XAI_lambda", default = 0.0, type= float)
parser.add_argument("--name", default = "", type=str)
parser.add_argument("--use_hidden_feature", action='store_true')
parser.add_argument("--batch_size", default =100)

# try:  # terminal
parser = parser.parse_args()
# except:  # Jupyter notebook
# parser = parser.parse_args(args=[])

train_dataset = ArgoverseDataset(parser.dataroot, mode='train', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

val_dataset = ArgoverseDataset(parser.dataroot, mode='val', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

test_dataset = ArgoverseDataset(parser.dataroot, mode='test', delta=parser.predict_delta,
                              map_features_flag=parser.map_features,
                              social_features_flag=True, heuristic=(not parser.no_heuristic),
                              ifc=parser.IFC, is_oracle=parser.use_oracle)

train_loader = DataLoader(train_dataset, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=True, drop_last=True)

val_loader = DataLoader(val_dataset, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)

test_loader = DataLoader(test_dataset, batch_size=parser.batch_size, num_workers=parser.workers,
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)


model = WIMP(parser)
model.load_state_dict(torch.load("experiments/example_old/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) # 학습할 때에는 graph 모듈에서 p에 해당하는 network가 없었으므로

model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

# conv_weight_origin = copy.deepcopy(model.module.decoder.xy_conv_filters[0].weight)
# last_weight_origin = copy.deepcopy(model.module.decoder.value_generator.weight)
conv_weight_origin = copy.deepcopy(model.decoder.xy_conv_filters[0].weight)
last_weight_origin = copy.deepcopy(model.decoder.value_generator.weight)


now = time.localtime()
start_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)
start_time = parser.name + str(parser.XAI_lambda) + start_time

save_foler = "ResultsImg/"
save_XAI = save_foler + "/XAI/"
save_attention = save_foler + "/attention"
save_folder = "results_XAI/" + start_time + "___" + str(parser.XAI_lambda).replace(".", "_") + "/"

os.mkdir(save_folder)
if parser.draw_image:
    os.mkdir("ResultsImg/" + start_time)

slicing = lambda a, idx: torch.cat((a[:idx], a[idx+1:]), axis=1)
slicing_2Dpadding = lambda a, idx: torch.cat((a[:idx], a[idx+1:], torch.zeros_like(a[0:1])), axis=0)
slicing_1Dpadding = lambda a, idx: F.pad(torch.cat((a[:idx], a[idx+1:]), axis=0), (0,1))


abs_min = lambda weight, k: torch.topk(abs(weight)[1:], k+1, largest = False).indices[k].item()
abs_max = lambda weight, k: torch.topk(abs(weight)[1:], k+1).indices[k].item()
simple_min = lambda weight, k: torch.topk(weight[1:], k+1, largest = False).indices[k].item()
simple_max = lambda weight, k: torch.topk(weight[1:], k+1).indices[k].item()



# +
num_socials = []

write_json_original = []
write_json_delete = [[[],[],[]] , [[],[],[]], [[],[],[]], [[],[],[]]]  #[names][delete_num][data_iter]

f_n = ["abs_min", "abs_max", "simple_min", "simple_max"]

save_metric = {
    "abs_min" : {
    "delete1" : XAI_utils_renewer.create_dict(),
    "delete2" : XAI_utils_renewer.create_dict(),
    "delete3" : XAI_utils_renewer.create_dict()
    },
    "abs_max" : {
    "delete1" : XAI_utils_renewer.create_dict(),
    "delete2" : XAI_utils_renewer.create_dict(),
    "delete3" : XAI_utils_renewer.create_dict()
    },
    "simple_min" : {
    "delete1" : XAI_utils_renewer.create_dict(),
    "delete2" : XAI_utils_renewer.create_dict(),
    "delete3" : XAI_utils_renewer.create_dict()
    },
    "simple_max" : {
    "delete1" : XAI_utils_renewer.create_dict(),
    "delete2" : XAI_utils_renewer.create_dict(),
    "delete3" : XAI_utils_renewer.create_dict()
    }
}

torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)
torch.backends.cudnn.benchmark = False
    
original_model_metric = XAI_utils_renewer.create_dict()
DA_model_metric = [[XAI_utils_renewer.create_dict() for _ in range(3)] for _ in range(4)] # 하나씩 지우면서 metric을 잴것임

    
for batch_idx, batch in enumerate(tqdm(val_loader)):
    
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
    
    # gradient 유지
    adjacency.retain_grad()
    gan_features.retain_grad()

    loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
    loss.backward()

    XAI_utils_renewer.get_metric(original_model_metric, ade, fde, mr, loss,len(adjacency.grad)) 

    adjacency_grad_raw = adjacency * adjacency.grad
    feature_grad_raw = torch.sum(gan_features.grad * gan_features, axis=3).squeeze(-1)

    optimizer.zero_grad()
    optimizer.step()


    if parser.remove_high_related_score:
        if parser.draw_image:
            write_dict = copy.deepcopy([XAI_utils_renewer.metric_to_dict(preds[i], waypoint_preds, input_dict, target_dict, 
                                                                         attention,i, adjacency_grad_raw[i][0]) for i in range(len(preds))])
            write_json_original += write_dict

        for name_idx, function in enumerate([abs_min, abs_max, simple_min, simple_max]): # 각 방법별로 하나씩 지우기
           
            adjacency_grad = adjacency_grad_raw.clone()
            feature_grad = feature_grad_raw.clone()
            input_dict_clone = copy.deepcopy(input_dict)
            target_dict_clone = copy.deepcopy(target_dict)
            
            for i in range(parser.maximum_delete_num):  # 하나씩 지우자
                remove_idx = [] # batch안의 iteration을 통째로 지울 것임
                for idx in range(len(input_dict_clone["agent_features"])): # batch 안의 iteration별로
                    num_vehicles =  int(torch.sum(input_dict_clone["num_agent_mask"][idx]))

                    if num_vehicles > 1: # social agent가 1이상일때, 즉 첫번째 social agent의 data가 존재하는 경우
                        if parser.use_hidden_feature:
                            arg = function(feature_grad[idx][:num_vehicles], 0)

                        else:
                            arg = function(adjacency_grad[idx][0][:num_vehicles], 0)
                            
                        feature_grad[idx] = slicing_1Dpadding(feature_grad[idx], arg+1) # feature_grad지움
                        adjacency_grad[idx][0] = slicing_1Dpadding(adjacency_grad[idx][0],arg+1)
                        input_dict_clone["num_agent_mask"][idx] = slicing_1Dpadding(input_dict_clone["num_agent_mask"][idx], arg+1)
                        input_dict_clone["social_features"][idx] = slicing_2Dpadding(input_dict_clone["social_features"][idx], arg) #  agent 지움
                        input_dict_clone["ifc_helpers"]["social_oracle_centerline"][idx] = slicing_2Dpadding(input_dict_clone["ifc_helpers"]["social_oracle_centerline"][idx], arg)
                        input_dict_clone["ifc_helpers"]["social_oracle_centerline_lengths"][idx] = slicing_2Dpadding(input_dict_clone["ifc_helpers"]["social_oracle_centerline_lengths"][idx],arg)
                        
                    else:
                        remove_idx.append(idx)
                
                preds_,waypoint_preds_, _,_,_,_,_= model(**input_dict_clone)
                
                preds_original = preds_.clone()
                waypoint_preds_original = [waypoint_preds_[0].clone(), waypoint_preds_[1].clone()]
                
                
                # evaluation할 때 지우지 않은것은 무시
                for d_id in remove_idx:
                    preds_  = torch.cat((preds_[:d_id], preds_[d_id+1:]), axis = 0) 
                    for iii in range(len(waypoint_preds_)):
                        waypoint_preds_[iii]  = torch.cat((waypoint_preds_[iii][:d_id], waypoint_preds_[iii][d_id+1:]), axis = 0)
                    target_dict_clone['agent_labels']  = torch.cat((target_dict_clone['agent_labels'][:d_id], target_dict_clone['agent_labels'][d_id+1:]), axis = 0)

                
                loss, (ade, fde, mr) = model.eval_preds(preds_, target_dict_clone, waypoint_preds_)
                XAI_utils_renewer.get_metric(DA_model_metric[name_idx][i], ade, fde, mr, loss, len(adjacency_grad) - len(remove_idx))
                target_dict_clone = copy.deepcopy(target_dict)
                if parser.draw_image:
                    write_dict = copy.deepcopy([XAI_utils_renewer.metric_to_dict(preds_original[j], waypoint_preds_original, input_dict_clone, 
                                                                                 target_dict, attention,j, adjacency_grad[j][0]) for j in range(len(preds_original))])
                    write_json_delete[name_idx][i] += write_dict

            waypoint_preds_original[0].detach()
            waypoint_preds_original[1].detach()
            target_dict_clone['agent_labels'].detach()
            
    adjacency_grad_raw.detach()
    feature_grad_raw.detach()
    torch.cuda.empty_cache()

        

for name_idx in range(4):
    save_metric[f_n[name_idx]]['delete1'] = XAI_utils_renewer.calc_mean(DA_model_metric[name_idx][0])
    save_metric[f_n[name_idx]]['delete2'] = XAI_utils_renewer.calc_mean(DA_model_metric[name_idx][1])
    save_metric[f_n[name_idx]]['delete3'] = XAI_utils_renewer.calc_mean(DA_model_metric[name_idx][2])

save_metric['original_model_metric'] = XAI_utils_renewer.calc_mean(original_model_metric)
save_metric['XAI_lambda'] = parser.XAI_lambda

print("SAVE : ", save_folder + "metrics.json")
with open(save_folder + "metrics.json", "w") as json_data:
    json.dump(save_metric, json_data, indent=4)

print(save_metric)

if False: # sanitc check
    for item in range(len(write_json_delete[0][0])):
        for d_num in range(len(write_json_delete[0])):
            for m_num in range(len(write_json_delete)):

                mask = write_json_delete[m_num][d_num][item]["mask"]
                social_features = write_json_delete[m_num][d_num][item]["social_features"]

                mask = np.sum(mask).item()
                social_features = np.sum(np.array(social_features)[:, :1] != 0)
                assert (mask*2 -2 == social_features ), (mask, social_features)

    for item in range(len(write_json_original)):
        mask = write_json_original[item]["mask"]
        social_features = write_json_original[item]["social_features"]
        mask = np.sum(mask).item()
        social_features = np.sum(np.array(social_features)[:, :1] != 0)
        assert (mask*2 -2 == social_features ), (mask, social_features)



if parser.draw_image:
    print("DRAW IMAGE !!!")
    for d in tqdm(range(len(write_json_original))):
        xyrange = None
        max_weight = max(np.max(abs(np.array(write_json_original[d]['weight']))), 0.00000001)
        for d_num in range(4):
            fig = plt.figure(figsize=(20, 20))
            
            for f in range(4):
                name = f"{f_n[f]}_____delete {d_num}____dataset {d}"
                if d_num == 0:
                    data = write_json_original[d]
                else:
                    data = write_json_delete[f][d_num-1][d]


                agent_features = np.array(data["agent_features"])
                social_features = np.array(data["social_features"])
                preds = np.array(data["preds"])[0]
                city = data["city"]
                rotation = data["rotation"]
                translation = data["translation"]
                draw_future = True, 
                weight = np.array(data["weight"])
                masking = data["mask"]
                gt = data["gt"]

                plt.subplot(221 + f)
                xyrange = XAI_utils_renewer.draw(agent_features, social_features, preds, city, rotation,translation, gt = gt,
                              draw_future = True, weight = weight, plot_name = name, plt = plt, social_nums = np.sum(masking)-1, 
                            xyrange = xyrange, max_weight = max_weight)
                plt.tight_layout()

            save_name = f"ResultsImg/{start_time}/dataset_{d}___delete_{d_num}.png"
            plt.savefig(save_name)
            plt.close()

if parser.make_submit_file:
    from argoverse.evaluation.competition_util import generate_forecasting_h5
    file_names = ["origial", "delete_one", "delete_two", "delete_three"]

    for data, output_path in zip([write_json_original, write_json_delete[0], write_json_delete[1], write_json_delete[2]], file_names):
        output_path = "competition_files/" + output_path + "/" + output_path + ".h5"

        if parser.make_submit_file:
            output_all = {}
            for d in tqdm(data):
                p, seq_id = XAI_utils_renewer.prediction(d)
                output_all[seq_id] = p
            generate_forecasting_h5(output_all, output_path)



# +
# # !zip -r image.zip  ResultsImg/{start_time}/
