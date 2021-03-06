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

# # +
from argoverse.map_representation.map_api import ArgoverseMap

am = ArgoverseMap()

# # +
import XAI_utils

import torch.backends.cudnn as cudnn
import random
os.environ["CUDA_VISIBLE_DEVICES"]= "2"


from argparse import ArgumentParser
parser = ArgumentParser()

# + endofcell="--"
def draw(agent_features, social_features, preds, city_name, rotation, translation, social_nums = None ,weight = None, gt = None,
             draw_future = True, figsize = (8,8), save_fig = False, save_name = None, plot_name = None, plt = None, xyrange = None, max_weight = None):
    
    
    if not max_weight:
        max_weight = max(np.max(abs(weight)), 0.000001)
    weight /= max_weight
    
    if plot_name:
        plt.title(plot_name)
    
    angle_ans = -rotation
    xmin, xmax, ymin,ymax = [9999], [-9999], [9999], [-9999]
    preds = [[preds[i] - preds[i][:1] for i in range(len(preds))]]    # calculate x's mean and y's mean in predict
    
    agent_features = XAI_utils.denormalization(agent_features, angle_ans, -translation[0], -translation[1])
    plt.plot(agent_features[...,0],agent_features[...,1], color='orange',linewidth=3)
    plt.scatter(agent_features[...,0][-1],agent_features[...,1][-1], color='orange',linewidth=8)
    plt.text(agent_features[...,0][-1],agent_features[...,1][-1], np.round(weight_original[0], 3))

    if draw_future:
        for p in preds[0]:
            p = XAI_utils.denormalization(p, angle_ans, -translation[0], -translation[1])
            p += agent_features[-1] - agent_features[0]
            plt.plot(p[...,0], p[...,1], linestyle='dashed', alpha = 0.5, zorder = -1)
            plt.scatter(p[...,0][-1], p[...,1][-1], linewidth=2, alpha = 0.5, zorder = -1)
            xmin, xmax = min(np.append(xmin, p[...,0])), max(np.append(xmax, p[...,0]))
            ymin, ymax = min(np.append(ymin, p[...,1])), max(np.append(ymax, p[...,1]))
    gt = np.array(gt)

    if np.sum(gt) > 0:
        p = XAI_utils.denormalization(gt, angle_ans, -translation[0], -translation[1])
        # p += agent_features[-1] - agent_features[0]
        plt.plot(p[...,0], p[...,1], color = "deepskyblue")
        plt.scatter(p[...,0][-1], p[...,1][-1], color = "deepskyblue", s=80, marker=(5, 1))
        xmin, xmax = min(min(np.append(xmin, p[...,0])), xmin), max(max(np.append(xmax, p[...,0])), xmax)
        ymin, ymax = min(min(np.append(ymin, p[...,1])), ymin), max(max(np.append(ymax, p[...,1])), ymax)
    
    for i, AV in enumerate(social_features):
        AV = XAI_utils.denormalization(np.array(AV), angle_ans, -translation[0], -translation[1])
        # if True:
        if i < social_nums: # ?????? AV??? trajectory??? 0?????? ????????? ?????? ?????? ????????? ????????? ??? ?????? AGENT??? ???????????? ????????? ???????????? ??????????????? ?????????
            #AV += AV[-1] - AV[0]
            plt.plot(AV[...,0],AV[...,1], color='black')
            plt.scatter(AV[...,0][-1],AV[...,1][-1], color='black')
            if weight[i+1] > 0:
                plt.scatter(AV[...,0][-1],AV[...,1][-1], color='red', linewidth = 8, alpha = weight[i+1])
            else:
                plt.scatter(AV[...,0][-1],AV[...,1][-1], color='blue', linewidth = 8, alpha = -weight[i+1])
            plt.text(AV[...,0][-1],AV[...,1][-1], np.round(weight_original[i+1], 3))
            xmin = min(np.append(np.append(agent_features[...,0], AV[...,0]),xmin))
            xmax = max(np.append(np.append(agent_features[...,0], AV[...,0]),xmax))
            ymin = min(np.append(np.append(agent_features[...,1], AV[...,1]),ymin))
            ymax = max(np.append(np.append(agent_features[...,1], AV[...,1]),ymax))
    
    if xyrange:
        pass
    else:
        xyrange = [xmin-20, xmax+20, ymin-20, ymax]
        
    local_lane_polygons = am.find_local_lane_polygons(xyrange, city_name)
    for l in local_lane_polygons:
        plt.plot(l[...,0],l[...,1], linewidth='0.5', color='gray')

    if save_fig:
        assert save_name, "For save figure save_name is required "
        plt.savefig(save_name)
        plt.close()
    else:
        pass
        print("SSSSSSS")
        plt.show()
    return xyrange

# -

args = {"IFC":True, "add_centerline":False, "attention_heads":4, "batch_norm":False, "batch_size":24, "check_val_every_n_epoch":3, 
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
model.load_state_dict(torch.load("experiments/example/checkpoints/epoch=122.ckpt")['state_dict'], strict=False) # ????????? ????????? graph ???????????? p??? ???????????? network??? ???????????????
# model =nn.parallel.DataParallel(model)

model = model.cuda()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

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

softmax = torch.nn.Softmax(dim=1)


# + endofcell="--"

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
    ]  # ????????? ???????????? metric??? ?????????
    
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
        
            
        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        loss.backward()
        assert adjacency.grad != None, "adjacency_grad is None"
        assert gan_features.grad != None, "gan_features_grad is None"

        get_metric(original_model_metric, ade, fde, mr, loss,len(adjacency.grad)) 
        optimizer.zero_grad()
        optimizer.step()
        adjacency_grad = adjacency * adjacency.grad
        feature_grad = torch.sum(gan_features.grad * gan_features, axis=3).squeeze(-1)
        
#         #  sanity check
#         conv_weight = model.decoder.xy_conv_filters[0].weight
#         last_weight = model.decoder.value_generator.weight
#         assert torch.all(conv_weight == conv_weight_origin) and torch.all(last_weight == last_weight_origin), ("Model is Changed",conv_weight,conv_weight_origin,last_weight,last_weight_origin,)

        if parser.draw_image: # batch??? ????????? ????????????. ??? ??? ??? input_dict??? ????????? ????????? ??????
            for idx in range(len(input_dict["agent_features"])):
                num_socials.append(int(torch.sum(input_dict["num_agent_mask"][idx]).item()) - 1)
                weight = adjacency.grad[idx][0].cpu().numpy()
                att = attention[idx].cpu().detach().numpy()
                agent_features = input_dict["agent_features"][idx].cpu().numpy()
                social_features = input_dict["social_features"][idx].cpu().numpy()
                # target = target_dict['agent_labels'][idx].cpu().numpy()
                batch_preds = preds[idx][:, :, :, :2][0].cpu().detach().numpy()
                city_name = input_dict["ifc_helpers"]["city"][idx]
                rotation = input_dict["ifc_helpers"]["rotation"][idx].numpy()
                translation = input_dict["ifc_helpers"]["translation"][idx].numpy()
#                 XAI_utils.draw_attention(agent_features,social_features,batch_preds,city_name,rotation,translation,weight=copy.deepcopy(att),draw_future=True,
#                     save_fig=True,save_name=save_attention+ "/"+ str(batch_idx)+ "_"+ str(idx)+ ".png",)
                XAI_utils.draw(agent_features,social_features, batch_preds, city_name, rotation, translation, weight=copy.deepcopy(weight), draw_future=True,
                    save_fig=True, save_name=save_XAI + "/" + str(batch_idx) + "_" + str(idx) + ".png")
                  


        if parser.remove_high_related_score:
            if parser.save_json:
                write_dict = copy.deepcopy([metric_to_dict(preds[i], waypoint_preds, input_dict, target_dict, attention,i, adjacency_grad[i][0]) for i in range(len(preds))])
                write_json_original += write_dict

            with torch.no_grad():
                for i in range(parser.maximum_delete_num):  # ????????? ?????????
                    remove_idx = []
                    for idx in range(len(input_dict["agent_features"])): # batch ?????? iteration??????
                        num_vehicles =  int(torch.sum(input_dict["num_agent_mask"][idx]))
                        if parser.use_hidden_feature:
                            weight = feature_grad[idx]
                        else:
                            weight = adjacency_grad[idx][0]
                        
                        if num_vehicles > 1: # social agent??? 1????????????, ??? ????????? social agent??? data??? ???????????? ??????
                            arg = function(weight[:num_vehicles], 0)
                            weight = torch.cat((weight[: arg + 1], weight[arg + 2 :]))

                            input_dict["num_agent_mask"][idx] = slicing_1Dpadding(input_dict["num_agent_mask"][idx], arg + 1)
                            adjacency_grad[idx][0] = slicing_1Dpadding(adjacency_grad[idx][0], arg+1) # adjacency ??????
                            feature_grad[idx] = slicing_1Dpadding(feature_grad[idx], arg+1) # feature_grad??????

                            input_dict["social_features"][idx] = slicing_2Dpadding(input_dict["social_features"][idx], arg) #  agent ??????
                            input_dict["ifc_helpers"]["social_oracle_centerline"][idx] = slicing_2Dpadding(input_dict["ifc_helpers"]["social_oracle_centerline"][idx], arg)
                            input_dict["ifc_helpers"]["social_oracle_centerline_lengths"][idx] = slicing_2Dpadding(input_dict["ifc_helpers"]["social_oracle_centerline_lengths"][idx],arg,)
                        else:
                            remove_idx.append(idx)

                            
                    
                    preds_,waypoint_preds_,_,_,_ , _, _= model(**input_dict)
                    
                    target_dict_ = copy.deepcopy(target_dict)
                    
                    for d_id in remove_idx:
                        preds_  = torch.cat((preds_[:d_id], preds_[d_id+1:]), axis = 0)
                        for iii in range(len(waypoint_preds_)):
                            waypoint_preds_[iii]  = torch.cat((waypoint_preds_[iii][:d_id], waypoint_preds_[iii][d_id+1:]), axis = 0)
                        target_dict_['agent_labels']  = torch.cat((target_dict_['agent_labels'][:d_id], target_dict_['agent_labels'][d_id+1:]), axis = 0)

                    loss, (ade, fde, mr) = model.eval_preds(preds_, target_dict_, waypoint_preds_)
                    get_metric(DA_model_metric[i], ade, fde, mr, loss, len(adjacency_grad) - len(remove_idx))

                    
                    
                    
                    if parser.save_json:
                        write_dict = copy.deepcopy([metric_to_dict(preds_[j], waypoint_preds_, input_dict, target_dict, attention,j, adjacency_grad[j][0]) for j in range(len(preds))])
                        write_json_delete[name_idx][i] += write_dict

        adjacency.detach()
        gan_features.detach()
#         assert False, "STOP"
#         torch.cuda.empty_cache()

    def calc_mean(metric):
        if metric["length"] == 0:
            metric["length"] += 0.000000000001
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
    original_model_metric = calc_mean(original_model_metric)

    write_json = {
        "DA_model_metric": DA_model_metric,
        "original_model_metric": original_model_metric,
        "adjacency_exp": adjacency_exp,
        "XAI_lambda": parser.XAI_lambda
    }

    print("SAVE : ", save_folder + names[name_idx] + ".json")
    with open(save_folder + names[name_idx]  + ".json", "w") as json_data:
        json.dump(write_json, json_data, indent=4)

    print(write_json)
    break

# -

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

# # +
f_n = ["abs_min", "abs_max", "simple_min", "simple_max"]
from matplotlib import pyplot as plt

if parser.draw_image:
    for d in tqdm(range(len(write_json_original)//4)):
        xyrange = None
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
                xyrange = draw(agent_features, social_features, preds, city, rotation,translation, gt = gt,
                              draw_future = True, weight = weight, plot_name = name, plt = plt, social_nums = np.sum(masking)-1, xyrange = xyrange)
                plt.tight_layout()

            save_name = f"ResultsImg/{start_time}/dataset_{d}___delete_{d_num}.png"
            plt.savefig(save_name)
            plt.close()

if parser.make_submit_file:
    file_names = ["origial", "delete_one", "delete_two", "delete_three"]

    for data, output_path in zip([write_json_original, write_json_delete[0], write_json_delete[1], write_json_delete[2]], file_names):
        output_path = "competition_files/" + output_path + "/" + output_path + ".h5"
        if parser.make_submit_file:
            def denormalization(arr, angle, translation):

                theta = (angle)/180*math.pi
                c, s = np.cos(theta), np.sin(theta)
                R = np.array(((c, -s), (s, c)))

                #rotate
                arr = np.array(R.dot(arr.transpose())).transpose()
                #translate
                arr += translation
            #     arr[...,0] += (AGENT[19,0] - AGENT[0,0])
            #     arr[...,1] += (AGENT[19,1] - AGENT[0,1])
                return arr    


            def prediction(json_dict):
                preds = np.array(json_dict['preds'][0])[...,:2]
                preds = np.array([denormalization(p, -json_dict['rotation'], -np.array(json_dict['translation'])) for p in preds])
                seq_id = int(json_dict['csv_file'].split('.')[0])
                return preds, seq_id    


            from tqdm.notebook import tqdm
            from argoverse.evaluation.competition_util import generate_forecasting_h5
            ##########################======================================#############################
    #         data = write_json_delete[2]
    #         output_path =  "competition_files/delete_three/"
            ##########################======================================#############################

            output_all = {}
            for d in tqdm(data):
                p, seq_id = prediction(d)
                output_all[seq_id] = p
            generate_forecasting_h5(output_all, output_path)


def prediction_draw(root_dir, file):
    json_dict = {}

    with open(root_dir + file, 'r') as json_data:
        json_dict = json.load(json_data)
        
    print(json_dict['csv_file'])
    print(json_dict.keys())
    preds = np.array(json_dict['preds'][0])[...,:2]
    preds = np.array([denormalization(p, -json_dict['rotation'], -np.array(json_dict['translation'])) for p in preds])
    seq_id = int(json_dict['csv_file'].split('.')[0])
    return preds, seq_id

# --


calc_mean(original_model_metric)

DA_model_metric

target_dict_

torch.sum(model.encoder.xy_conv_filters[2].weight.grad)

adjacency.grad

# {'DA_model_metric': None, 'original_model_metric': {'fde': 1.1859465789794923, 'ade': 0.7981241607666015, 'loss': 5.974931640625, 'mr': 0.09999999046325683}, 'adjacency_exp': {'fde': 1.3464303588867188, 'ade': 0.8303599548339844, 'loss': 6.147362060546875, 'mr': 0.14}, 'XAI_lambda': 0.7}


torch.cuda.empty_cache()

graph_output.shape

gan_features.shape

adjacency_grad.shape

a = torch.tensor([[1,2,3,4,5]]).float()
softmax(adjacency_grad)

softmax(adjacency_grad)

m = nn.Softmax2d()
input = torch.randn(1,1,2, 3)
m(input) * len(input)

softmax = torch.nn.Softmax(dim=1)


m(adjacency_grad)
