# -*- coding: utf-8 -*-
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
import torch
import copy

from src.data.argoverse_datamodule import ArgoverseDataModule
from src.data.argoverse_dataset import ArgoverseDataset
from src.data.dummy_datamodule import DummyDataModule
from src.models.WIMP import WIMP

from torch.utils.data import DataLoader, Dataset

# +
import XAI_utils

import torch.backends.cudnn as cudnn
import random

# -

args = {"IFC":True, "add_centerline":False, "attention_heads":4, "batch_norm":False, "batch_size":1, "check_val_every_n_epoch":3, 
          "dataroot":'./data/argoverse_processed_simple', "dataset":'argoverse', "distributed_backend":'ddp', "dropout":0.0, 
          "early_stop_threshold":5, "experiment_name":'example', "gpus":3, "gradient_clipping":True, "graph_iter":1, 
          "hidden_dim":512, "hidden_key_generator":True, "hidden_transform":False, "input_dim":2, "k_value_threshold":10, 
          "k_values":[6, 5, 4, 3, 2, 1], "lr":0.0001, "map_features":False, "max_epochs":200, "mode":'train', "model_name":'WIMP', 
          "no_heuristic":False, "non_linearity":'relu', "num_layers":4, "num_mixtures":6, "num_nodes":1, "output_conv":True, "output_dim":2, 
          "output_prediction":True, "precision":32, "predict_delta":False, "resume_from_checkpoint":None, 
          "scheduler_step_size":[60, 90, 120, 150, 180], "seed":None, "segment_CL":False, "segment_CL_Encoder":False, 
          "segment_CL_Encoder_Gaussian":False, "segment_CL_Encoder_Gaussian_Prob":False, "segment_CL_Encoder_Prob":True, 
          "segment_CL_Gaussian_Prob":False, "segment_CL_Prob":False, "use_centerline_features":True, "use_oracle":False, "waypoint_step":5, 
          "weight_decay":0.0, "workers":8, "wta":False, "draw_image" : False, "remove_high_related_score" : True, "maximum_delete_num" : 3}



train_loader = ArgoverseDataset(args['dataroot'], mode='train', delta=args['predict_delta'],
                              map_features_flag=args['map_features'],
                              social_features_flag=True, heuristic=(not args['no_heuristic']),
                              ifc=args['IFC'], is_oracle=args['use_oracle'])

val_loader = ArgoverseDataset(args['dataroot'], mode='val', delta=args['predict_delta'],
                              map_features_flag=args['map_features'],
                              social_features_flag=True, heuristic=(not args['no_heuristic']),
                              ifc=args['IFC'], is_oracle=args['use_oracle'])

test_loader = ArgoverseDataset(args['dataroot'], mode='test', delta=args['predict_delta'],
                              map_features_flag=args['map_features'],
                              social_features_flag=True, heuristic=(not args['no_heuristic']),
                              ifc=args['IFC'], is_oracle=args['use_oracle'])

train_dataset = DataLoader(train_loader, batch_size=args['batch_size'], num_workers=args['workers'],
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=True, drop_last=True)

val_dataset = DataLoader(val_loader, batch_size=args['batch_size'], num_workers=args['workers'],
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)

trest_dataset = DataLoader(test_loader, batch_size=args['batch_size'], num_workers=args['workers'],
                                pin_memory=True, collate_fn=ArgoverseDataset.collate,
                                shuffle=False, drop_last=False)
# dm = ArgoverseDataset(args)



# +
model = WIMP(args)
model.load_state_dict(torch.load("experiments/example/checkpoints/epoch=122.ckpt")['state_dict'])
model = model.cuda()

conv_weight_origin = copy.deepcopy(model.decoder.xy_conv_filters[0].weight)
last_weight_origin = copy.deepcopy(model.decoder.value_generator.weight)

# -

def get_metric(metric_dict, ade,fde,mr,loss):
    metric_dict["ade"] += ade
    metric_dict["fde"] += fde
    metric_dict["mr"] += mr
    metric_dict["loss"] += loss
    metric_dict["length"]+=1

#  optimizer는 선언하지 않았습니다.
#  따라서 아마 model weight에 gradient는 계속 쌓이겠지만 저희가 중요한 것은 adjacency matrix의 graident이므로 상관 없을 것으로 예측됩니다.
import os
import math
import copy
import torch.nn as nn
Relu = nn.ReLU()

save_foler = "ResultsImg/"

# +
save_XAI = save_foler + "/XAI/"
save_attention = save_foler + "/attention"

slicing = lambda a, idx: torch.cat((a[:, :idx], a[:, idx+1:]), axis=1)
# -

import os
os.listdir("results_XAI")

# +
import time

now = time.localtime()
 
start_time = "%02d_%02d_%02d_%02d" % (now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min)


# +
abs_min = lambda weight: torch.argmin(abs(weight)[1:]).item()
abs_max = lambda weight: torch.argmax(abs(weight)[1:]).item()
simple_min = lambda weight: torch.argmin(weight[1:]).item()
simple_max = lambda weight: torch.argmax(weight[1:]).item()


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

    original_model_metric = {
        "ade": torch.FloatTensor([0]).cuda(),
        "fde": torch.FloatTensor([0]).cuda(),
        "mr": torch.FloatTensor([0]).cuda(),
        "loss": torch.FloatTensor([0]).cuda(),
        "length": 0,
    }

    DA_model_metric = [
        {
            "ade": torch.FloatTensor([0]).cuda(),
            "fde": torch.FloatTensor([0]).cuda(),
            "mr": torch.FloatTensor([0]).cuda(),
            "loss": torch.FloatTensor([0]).cuda(),
            "length": 0,
        },
        {
            "ade": torch.FloatTensor([0]).cuda(),
            "fde": torch.FloatTensor([0]).cuda(),
            "mr": torch.FloatTensor([0]).cuda(),
            "loss": torch.FloatTensor([0]).cuda(),
            "length": 0,
        },
        {
            "ade": torch.FloatTensor([0]).cuda(),
            "fde": torch.FloatTensor([0]).cuda(),
            "mr": torch.FloatTensor([0]).cuda(),
            "loss": torch.FloatTensor([0]).cuda(),
            "length": 0,
        },
    ]  # 하나씩 지우면서 metric을 잴것임

    for batch_idx, batch in enumerate(tqdm(val_dataset)):
        input_dict, target_dict = batch[0], batch[1]

        # get cuda
        input_dict["agent_features"] = input_dict["agent_features"].cuda()
        input_dict["social_features"] = input_dict["social_features"].cuda()
        input_dict["social_label_features"] = input_dict["social_label_features"].cuda()
        input_dict["adjacency"] = input_dict["adjacency"].cuda()
        input_dict["label_adjacency"] = input_dict["label_adjacency"].cuda()
        input_dict["num_agent_mask"] = input_dict["num_agent_mask"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict[
            "ifc_helpers"
        ]["agent_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"] = input_dict[
            "ifc_helpers"
        ]["agent_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline"] = input_dict[
            "ifc_helpers"
        ]["social_oracle_centerline"].cuda()
        input_dict["ifc_helpers"]["social_oracle_centerline_lengths"] = input_dict[
            "ifc_helpers"
        ]["social_oracle_centerline_lengths"].cuda()
        input_dict["ifc_helpers"]["agent_oracle_centerline"] = input_dict[
            "ifc_helpers"
        ]["agent_oracle_centerline"].cuda()
        target_dict["agent_labels"] = target_dict["agent_labels"].cuda()
        #         if batch_idx == 0:
        #             print(input_dict['social_features'][0,:,:2,:2])

        input_dict["adjacency"].requires_grad = True
        input_dict["adjacency"].retain_grad()

        preds, waypoint_preds, all_dist_params, attention, adjacency = model(
            **input_dict
        )
        adjacency.retain_grad()
        loss, (ade, fde, mr) = model.eval_preds(preds, target_dict, waypoint_preds)
        get_metric(original_model_metric, ade, fde, mr, loss)

        loss.backward()
        batch_preds = preds

        conv_weight = model.decoder.xy_conv_filters[0].weight
        last_weight = model.decoder.value_generator.weight

        assert torch.all(conv_weight == conv_weight_origin) and torch.all(
            last_weight == last_weight_origin
        ), (
            "Model is Changed",
            conv_weight,
            conv_weight_origin,
            last_weight,
            last_weight_origin,
        )

        for idx in range(args["batch_size"]):

            if args["draw_image"]:
                weight = adjacency.grad[idx][0].cpu().numpy()
                att = attention[idx].cpu().numpy()
                agent_features = input_dict["agent_features"][idx].cpu().numpy()
                social_features = input_dict["social_features"][idx].cpu().numpy()
                # target = target_dict['agent_labels'][idx].cpu().numpy()
                preds = batch_preds[idx][:, :, :, :2][0].cpu().detach().numpy()
                city_name = input_dict["ifc_helpers"]["city"][idx]
                rotation = input_dict["ifc_helpers"]["rotation"][idx].numpy()
                translation = input_dict["ifc_helpers"]["translation"][idx].numpy()
                XAI_utils.draw_attention(
                    agent_features,
                    social_features,
                    preds,
                    city_name,
                    rotation,
                    translation,
                    weight=copy.deepcopy(att),
                    draw_future=True,
                    save_fig=True,
                    save_name=save_attention
                    + "/"
                    + str(batch_idx)
                    + "_"
                    + str(idx)
                    + ".png",
                )

                XAI_utils.draw(
                    agent_features,
                    social_features,
                    preds,
                    city_name,
                    rotation,
                    translation,
                    weight=copy.deepcopy(weight),
                    draw_future=True,
                    save_fig=True,
                    save_name=save_XAI + "/" + str(batch_idx) + "_" + str(idx) + ".png",
                )

            if args["remove_high_related_score"]:
                weight = adjacency.grad[idx][0]
                for i in range(args["maximum_delete_num"]):
                    #                 print(batch_idx, input_dict["social_features"].shape)
                    if len(input_dict["social_features"][0] > 1):
                        arg = function(weight)
                        #                     arg_max = torch.argmin(weight[1:]).item()

                        weight = torch.cat((weight[: arg + 1], weight[arg + 2 :]))
                        input_dict["social_features"] = slicing(
                            input_dict["social_features"], arg
                        )
                        input_dict["num_agent_mask"] = slicing(
                            input_dict["num_agent_mask"], arg + 1
                        )
                        input_dict["ifc_helpers"]["social_oracle_centerline"] = slicing(
                            input_dict["ifc_helpers"]["social_oracle_centerline"], arg
                        )
                        input_dict["ifc_helpers"][
                            "social_oracle_centerline_lengths"
                        ] = slicing(
                            input_dict["ifc_helpers"][
                                "social_oracle_centerline_lengths"
                            ],
                            arg,
                        )
                    else:
                        break

                    with torch.no_grad():
                        (
                            preds,
                            waypoint_preds,
                            all_dist_params,
                            att_weights,
                            adjacency,
                        ) = model(
                            input_dict["agent_features"],
                            input_dict["social_features"],
                            None,
                            input_dict["num_agent_mask"],
                            ifc_helpers={
                                "social_oracle_centerline": input_dict["ifc_helpers"][
                                    "social_oracle_centerline"
                                ],
                                "social_oracle_centerline_lengths": input_dict[
                                    "ifc_helpers"
                                ]["social_oracle_centerline_lengths"],
                                "agent_oracle_centerline": input_dict["ifc_helpers"][
                                    "agent_oracle_centerline"
                                ],
                                "agent_oracle_centerline_lengths": input_dict[
                                    "ifc_helpers"
                                ]["agent_oracle_centerline_lengths"],
                            },
                        )
                        loss, (ade, fde, mr) = model.eval_preds(
                            preds, target_dict, waypoint_preds
                        )
                        get_metric(DA_model_metric[i], ade, fde, mr, loss)
    import json

    def calc_mean(metric):
        metric["fde"] /= metric["length"]
        metric["ade"] /= metric["length"]
        metric["mr"] /= metric["length"]
        metric["loss"] /= metric["length"]
        return {
            "fde": metric["fde"].item(),
            "ade": metric["ade"].item(),
            "loss": metric["loss"].item(),
            "mr": metric["mr"].item(),
        }

    for i in range(len(DA_model_metric)):
        DA_model_metric[i] = calc_mean(DA_model_metric[i])
    original_model_metric = calc_mean(original_model_metric)

    write_json = {
        "DA_model_metric": DA_model_metric,
        "original_model_metric": original_model_metric,
    }

    file_name = start_time + names[name_idx]
    print("SAVE : ", "results_XAI/" + file_name + ".json")
    with open("results_XAI/" + file_name + ".json", "w") as json_data:
        json.dump(write_json, json_data, indent=4)

    print(write_json)

# +
import os

os.listdir("results_XAI")
