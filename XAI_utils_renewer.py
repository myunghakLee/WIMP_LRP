# -*- coding: utf-8 -*-
# +
from argoverse.map_representation.map_api import ArgoverseMap
from tqdm.notebook import tqdm
import numpy as np
import math
import copy
am = ArgoverseMap()

def draw(agent_features, social_features, preds, city_name, rotation, translation, social_nums = None ,weight = None, gt = None,
             draw_future = True, figsize = (8,8), save_fig = False, save_name = None, plot_name = None, plt = None, xyrange = None, max_weight = None):
    
    weight_original = copy.deepcopy(weight)
    if not max_weight:
        max_weight = max(np.max(abs(weight)), 0.000001)
    weight /= max_weight

    
    if plot_name:
        plt.title(plot_name)
    
    angle_ans = -rotation
    xmin, xmax, ymin,ymax = [9999], [-9999], [9999], [-9999]
    preds = [[preds[i] - preds[i][:1] for i in range(len(preds))]]    # calculate x's mean and y's mean in predict
    
    agent_features = denormalization(agent_features, angle_ans, -translation[0], -translation[1])
    plt.plot(agent_features[...,0],agent_features[...,1], color='orange',linewidth=3)
    plt.scatter(agent_features[...,0][-1],agent_features[...,1][-1], color='orange',linewidth=8)
    plt.text(agent_features[...,0][-1],agent_features[...,1][-1], np.round(weight_original[0], 3))

    if draw_future:
        for p in preds[0]:
            p = denormalization(p, angle_ans, -translation[0], -translation[1])
            p += agent_features[-1] - agent_features[0]
            plt.plot(p[...,0], p[...,1], linestyle='dashed', alpha = 0.5, zorder = -1)
            plt.scatter(p[...,0][-1], p[...,1][-1], linewidth=2, alpha = 0.5, zorder = -1)
            xmin, xmax = min(np.append(xmin, p[...,0])), max(np.append(xmax, p[...,0]))
            ymin, ymax = min(np.append(ymin, p[...,1])), max(np.append(ymax, p[...,1]))
    gt = np.array(gt)

    if np.sum(gt) > 0:
        p = denormalization(gt, angle_ans, -translation[0], -translation[1])
        # p += agent_features[-1] - agent_features[0]
#         plt.plot(p[...,0], p[...,1], color = "deepskyblue")
        plt.scatter(p[...,0][-1], p[...,1][-1], color = "red", s=128, marker=(5, 1))
        xmin, xmax = min(min(np.append(xmin, p[...,0])), xmin), max(max(np.append(xmax, p[...,0])), xmax)
        ymin, ymax = min(min(np.append(ymin, p[...,1])), ymin), max(max(np.append(ymax, p[...,1])), ymax)
    
    for i, AV in enumerate(social_features):
        AV = denormalization(np.array(AV), angle_ans, -translation[0], -translation[1])
        # if True:
        if i < social_nums: # 간혹 AV의 trajectory가 0으로 초기화 되어 있는 경우가 있는데 이 경우 AGENT의 초기값과 같아져 발생하는 오류때문에 넣어줌
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
#         plt.show()
    return xyrange



# -

def get_args(IFC=True, add_centerline=False, attention_heads=4, batch_norm=False, batch_size=24, check_val_every_n_epoch=3, 
          dataroot='./data/argoverse_processed_simple', dataset='argoverse', distributed_backend='ddp', dropout=0.0, 
          early_stop_threshold=5, experiment_name='example', gpus=3, gradient_clipping=True, graph_iter=1, 
          hidden_dim=512, hidden_key_generator=True, hidden_transform=False, input_dim=2, k_value_threshold=10, 
          k_values=[6, 5, 4, 3, 2, 1], lr=0.0001, map_features=False, max_epochs=200, mode='train', model_name='WIMP', 
          no_heuristic=False, non_linearity='relu', num_layers=4, num_mixtures=6, num_nodes=1, output_conv=True, output_dim=2, 
          output_prediction=True, precision=32, predict_delta=False, resume_from_checkpoint=None, 
          scheduler_step_size=[60, 90, 120, 150, 180], seed=None, segment_CL=False, segment_CL_Encoder=False, 
          segment_CL_Encoder_Gaussian=False, segment_CL_Encoder_Gaussian_Prob=False, segment_CL_Encoder_Prob=True, 
          segment_CL_Gaussian_Prob=False, segment_CL_Prob=False, use_centerline_features=True, use_oracle=False, waypoint_step=5, 
          weight_decay=0.0, workers=8, wta=False, draw_image = False, remove_high_related_score = True, maximum_delete_num = 3, 
          save_json= False, make_submit_file = False, use_hidden_feature = True, is_LRP= True, adjacency_exp = True):
    
    return {"IFC":IFC, "add_centerline":add_centerline, "attention_heads":attention_heads, "batch_norm":batch_norm, 
            "check_val_every_n_epoch":check_val_every_n_epoch, "dataroot":dataroot, "dataset":dataset, "distributed_backend":distributed_backend, 
            "dropout":dropout, "early_stop_threshold":early_stop_threshold, "experiment_name":'experiment_name', "gpus":gpus, 
            "gradient_clipping":gradient_clipping, "graph_iter":graph_iter, "hidden_dim":hidden_dim, "hidden_key_generator":hidden_key_generator, 
            "hidden_transform":hidden_transform, "input_dim":input_dim, "k_value_threshold":k_value_threshold, "k_values":k_values, "lr":lr, 
            "map_features":map_features, "max_epochs":max_epochs, "mode":'mode', "model_name":'model_name', "no_heuristic":no_heuristic, 
            "non_linearity":'non_linearity', "num_layers":num_layers, "num_mixtures":num_mixtures, "num_nodes":num_nodes, "output_conv":output_conv, 
            "output_dim":output_dim, "output_prediction":output_prediction, "precision":precision, "predict_delta":predict_delta, 
            "resume_from_checkpoint":resume_from_checkpoint, "scheduler_step_size":scheduler_step_size, "seed":seed, 
            "segment_CL":segment_CL, "segment_CL_Encoder":segment_CL_Encoder, "segment_CL_Encoder_Gaussian":segment_CL_Encoder_Gaussian, 
            "segment_CL_Encoder_Gaussian_Prob":segment_CL_Encoder_Gaussian_Prob, "segment_CL_Encoder_Prob":segment_CL_Encoder_Prob, 
            "segment_CL_Gaussian_Prob":segment_CL_Gaussian_Prob, "segment_CL_Prob":segment_CL_Prob, 
            "use_centerline_features":use_centerline_features, "use_oracle":use_oracle, "waypoint_step":waypoint_step, 
            "weight_decay":weight_decay, "workers":workers, "wta":wta, "draw_image" : draw_image, "remove_high_related_score" : remove_high_related_score, 
            "maximum_delete_num" : maximum_delete_num, "save_json": save_json, "make_submit_file" : make_submit_file, 
             "is_LRP": is_LRP, "adjacency_exp" : adjacency_exp}


def get_metric(metric_dict, ade,fde,mr,loss, length):
    metric_dict["ade"] += (ade * length).cpu().item()
    metric_dict["fde"] += (fde * length).cpu().item()
    metric_dict["mr"] += (mr * length).cpu().item()
    metric_dict["loss"] += (loss * length).cpu().item()
    metric_dict["length"]+=length


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



def create_dict():
    return {
        "ade": 0.0,
        "fde": 0.0,
        "mr": 0.0,
        "loss": 0.0,
        "length": 0,
    }


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


def prediction(json_dict):
    preds = np.array(json_dict['preds'][0])[...,:2]
    preds = np.array([denormalization(p, -json_dict['rotation'], -np.array(json_dict['translation'])) for p in preds])
    seq_id = int(json_dict['csv_file'].split('.')[0])
    return preds, seq_id    


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

