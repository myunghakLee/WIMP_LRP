# -*- coding: utf-8 -*-
# + endofcell="--"
# -*- coding: utf-8 -*-
import json

import pytorch_lightning as pl
import torch
import torch.nn as nn

from argparse import ArgumentParser
from pytorch_lightning.metrics.functional import accuracy
from torch.nn import functional as F

# # +
from src.models.WIMP_decoder import WIMPDecoder
from src.models.WIMP_encoder import WIMPEncoder
from src.models.GAT import GraphAttentionLayer, GraphAttentionLayerLRP

from src.util.metrics import compute_metrics
from src.util.loss import l1_ewta_loss, l1_ewta_loss_prob, l1_ewta_waypoint_loss,\
    l1_ewta_encoder_waypoint_loss
# -

import XAI_utils
import json



# # +
class WIMP(pl.LightningModule):
    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)

        # Configuration Params
        parser.add_argument("--hidden-dim", type=int, default=512, help=" hidden dimension")
        parser.add_argument("--input-dim", type=int, default=2, help=" input dimension")
        parser.add_argument("--output-dim", type=int, default=2, help=" output dimension")
        parser.add_argument("--graph-iter", type=int, default=1, help="Number of iterations of graph message passing") # NOQA
        parser.add_argument("--attention-heads", type=int, default=4, help="Number of GAT attention heads") # NOQA
        parser.add_argument("--num-layers", type=int, default=4, help="Number of RNN layers to use")
        parser.add_argument("--hidden-transform", action='store_true', help="Use hidden state as input to GAT") # NOQA
        parser.add_argument("--use-centerline-features", action='store_true', help="Use Centerline Features") # NOQA
        parser.add_argument('--num-mixtures', type=int, default=1, help="Number of mixtures to use in decoder") # NOQA
        parser.add_argument("--output-prediction", action='store_true', help="Use output instead of hidden state for prediction") # NOQA
        parser.add_argument("--output-conv", action='store_true', help="Use conv filters during prediction") # NOQA
        parser.add_argument("--non-linearity", type=str, default='tanh', help="Non Linearity to use")
        parser.add_argument("--batch-norm", action='store_true', help="Use batch norm")
        parser.add_argument("--hidden-key-generator", action='store_true', help="Use hidden state for key generation") # NOQA
        parser.add_argument("--add-centerline", action='store_true', help="Add centerline features to current XY features") # NOQA
        parser.add_argument("--waypoint-step", type=int, default=5, help="Timesteps between waypoints")
        parser.add_argument("--segment-CL", action='store_true', help="Use CL segment")
        parser.add_argument("--segment-CL-Encoder", action='store_true', help="Use CL segment encoder")
        parser.add_argument("--segment-CL-Encoder-Gaussian", action='store_true', help="Use CL segment encoder with gaussian attention") # NOQA
        parser.add_argument("--segment-CL-Prob", action='store_true', help="Use CL segment with prob prediction") # NOQA
        parser.add_argument("--segment-CL-Encoder-Prob", action='store_true', help="Use CL segment encoder with distance with prob") # NOQA
        parser.add_argument("--segment-CL-Encoder-Gaussian-Prob", action='store_true', help="Use CL segment encoder with gaussian attention with prob") # NOQA
        parser.add_argument("--segment-CL-Gaussian-Prob", action='store_true', help="Use CL segment with gaussian attention with prob") # NOQA

        # Optimization Params
        parser.add_argument("--lr", type=float, default=0.0001, help='Learning rate')
        parser.add_argument("--weight-decay", type=float, default=0.0, help="Weight Decay")
        parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate for training")
        parser.add_argument("--k-value-threshold", type=int, default=5, help="Threshold for k reduction")
        parser.add_argument("--k-values", nargs='+', default=[6, 5, 4, 3, 2, 1])
        parser.add_argument("--gradient-clipping", action='store_true', help="Enable gradient clipping")
        parser.add_argument("--scheduler-step-size", nargs='+', type=int, default=[30, 60, 90, 120, 150])
        parser.add_argument("--wta", action='store_true', help="Use Winner Takes All approach")

        return parser

    def __init__(self, hparams, hidden_dim=128):
        super(WIMP, self).__init__()
        self.hparams = hparams

        self.encoder = WIMPEncoder(self.hparams)
        if self.hparams.is_LRP:
            self.gat = GraphAttentionLayerLRP(self.hparams.hidden_dim, self.hparams.hidden_dim,
                                           self.hparams.graph_iter, self.hparams.attention_heads,
                                           self.hparams.dropout, XAI_lambda = self.hparams.XAI_lambda)
        else:
            self.gat = GraphAttentionLayer(self.hparams.hidden_dim, self.hparams.hidden_dim,
                                           self.hparams.graph_iter, self.hparams.attention_heads,
                                           self.hparams.dropout)
            
        self.decoder = WIMPDecoder(self.hparams)
        self.slicing = lambda a, idx: torch.cat((a[:, :idx], a[:, idx+1:]), axis=1)
        
        self.json_data = []
        self.softmax = torch.nn.Softmax(dim=2)
        
    def forward(self, agent_features, social_features, adjacency, num_agent_mask, outsteps=30,
                social_label_features=None, label_adjacency=None, classmate_forcing=True,
                labels=None, ifc_helpers=None, test=False, map_estimate=False,
                gt=None, idx=None, sample_next=False, num_predictions=1, am=None):
        # Encode agent and social features
        encoding, hidden, waypoint_predictions = self.encoder(agent_features, social_features,
                                                              num_agent_mask, ifc_helpers)

        waypoint_predictions_tensor_encoder = waypoint_predictions.squeeze(-2).view(
            agent_features.size(0), social_features.size(1) + 1, -1, agent_features.size(2))

        # Perform graph message passing
        if self.hparams.hidden_transform:
            gan_features = torch.cat(hidden, dim=0).transpose(0, 1).view(
                agent_features.size(0), social_features.size(1) + 1,
                hidden[0].size(0) * 2, hidden[0].size(2))
        else:
            gan_features = encoding.view(social_features.size(0), social_features.size(1) + 1,
                                         1, -1)
#             if gan_features.requires_grad:
#                 gan_features.retain_grad()
        if len(adjacency.shape) == 4:
            adjacency = torch.ones(gan_features.size(1), gan_features.size(1)).to(
                gan_features.get_device()).float().unsqueeze(0).repeat(gan_features.size(0), 1, 1)
    #         adjacency.retain_grad()  # backpropagation??? gradient ????????? ?????? ????????? ????????????

        adjacency.requires_grad = True  # backpropagation??? gradient ????????? ?????? ????????? ???????????? 
        adjacency = adjacency * num_agent_mask.unsqueeze(1) * num_agent_mask.unsqueeze(2)
        
        graph_output, att_weights = self.gat(gan_features, adjacency)  # att_weights??? LRP????????? ????????? ??????, ???????????? ???????????? ?????? attention
#         att_weights.requires_grad = True
#         att_weights.retain_grad()
        graph_output = graph_output.narrow(1, 0, 1).squeeze(1)
        if self.hparams.batch_norm:
            graph_output = self.encoding_bn(graph_output.transpose(1, 2).contiguous())
            graph_output = graph_output.transpose(1, 2).contiguous() # it may be wrong
        if self.hparams.hidden_transform:
            hidden_decoder = torch.chunk(graph_output.view(-1, self.hparams.num_layers*2, self.hparams.hidden_dim).transpose(0,1).contiguous(), 2, dim=0)
        else:
            hidden_decoder = (graph_output.view(-1, 1, self.hparams.hidden_dim).transpose(0,1).contiguous(), graph_output.view(-1, 1, self.hparams.hidden_dim).transpose(0,1).contiguous())
            hidden_decoder = (hidden_decoder[0].repeat(self.hparams.num_layers, 1, 1), hidden_decoder[1].repeat(self.hparams.num_layers,1,1))

        # Decode prediction
        decoder_input_features = agent_features.narrow(dim=1, start=agent_features.size(1)-1, length=1)
        last_n_predictions = []
        for i in self.hparams.xy_kernel_list:
            last_n_predictions.append(agent_features.narrow(dim=1, start=agent_features.size(1)-i, length=i))
        prediction_tensor, waypoints_prediction_tensor, prediction_stats = self.decoder(decoder_input_features, last_n_predictions, hidden_decoder, outsteps, ifc_helpers=ifc_helpers, sample_next=sample_next, map_estimate=map_estimate)

        return prediction_tensor, [waypoints_prediction_tensor, waypoint_predictions_tensor_encoder], prediction_stats, att_weights, adjacency, gan_features, graph_output

    def training_step(self, batch, batch_idx):
        # Compute predictions
        input_dict, target_dict = batch
        input_dict['adjacency'] = self.softmax(input_dict["adjacency"]) * len(input_dict['adjacency'][0])
        preds, waypoint_preds, all_dist_params, _, adjacency, _, _ = self(**input_dict)
        # Compute loss and metrics
        loss, metrics = self.eval_preds(preds, target_dict, waypoint_preds)
        agent_mean_ade, agent_mean_fde, agent_mean_mr = metrics
        
        
        pl.TrainResult(loss)

        # Log results from training step
        result = pl.TrainResult(loss)
        result.log('train/loss', loss, on_epoch=True, sync_dist=True)
        result.log('train/ade', agent_mean_ade, on_epoch=True, sync_dist=True)
        result.log('train/fde', agent_mean_fde, on_epoch=True, sync_dist=True)
        result.log('train/mr', agent_mean_mr, on_epoch=True, sync_dist=True)
        return result

    def validation_step(self, batch, batch_idx):
        # Compute predictions
        input_dict, target_dict = batch
        input_dict['adjacency'] = self.softmax(input_dict["adjacency"]) * len(input_dict['adjacency'][0])

        preds, waypoint_preds, all_dist_params, _, adjacency, _, _ = self(**input_dict)

        # Compute loss and metrics
        loss, metrics = self.eval_preds(preds, target_dict, waypoint_preds)
        agent_mean_ade, agent_mean_fde, agent_mean_mr = metrics

        # Log results from validation step
        result = pl.EvalResult(checkpoint_on=loss, early_stop_on=loss)
        result.log('val/loss', loss, on_epoch=True, sync_dist=True)
        result.log('val/ade', agent_mean_ade, on_epoch=True, sync_dist=True)
        result.log('val/fde', agent_mean_fde, on_epoch=True, sync_dist=True)
        result.log('val/mr', agent_mean_mr, on_epoch=True, sync_dist=True)
        return result
    
    def test_step(self, batch, batch_idx):
        # Compute predictions
        result = pl.EvalResult()
        input_dict, target_dict = batch
        preds, waypoint_preds, all_dist_params, attention, adjacency, _, _ = self(**input_dict)
        loss, (agent_mean_ade, agent_mean_fde, agent_mean_mr) = self.eval_preds(preds, target_dict, waypoint_preds)
        
        if self.hparams.is_valid:
            
            result.log('loss/test_all', loss)
            result.log('ade/test_all', agent_mean_ade)
            result.log('fde/test_all', agent_mean_fde)
            result.log('mr/test_all', agent_mean_mr)


            if False:            
                for idx in range(len(input_dict["social_features"][0])):
                    preds, waypoint_preds, all_dist_params, att_weights, adjacency, _ = self(input_dict["agent_features"],
                                                self.slicing(input_dict["social_features"], idx),
                                                None,
                                                self.slicing(input_dict["num_agent_mask"], idx),
                                                ifc_helpers ={
                                                    "social_oracle_centerline": self.slicing(input_dict["ifc_helpers"]['social_oracle_centerline'], idx), 
                                                    "social_oracle_centerline_lengths": self.slicing(input_dict["ifc_helpers"]['social_oracle_centerline_lengths'], idx),
                                                    "agent_oracle_centerline": input_dict["ifc_helpers"]["agent_oracle_centerline"],
                                                    "agent_oracle_centerline_lengths": input_dict["ifc_helpers"]["agent_oracle_centerline_lengths"]
                                                })
                    loss, (agent_mean_ade, agent_mean_fde, agent_mean_mr) = self.eval_preds(preds, target_dict, waypoint_preds)
                    result.log(f'loss/test_{str(idx).zfill(2)}', loss)
                    result.log(f'ade/test_{str(idx).zfill(2)}', agent_mean_ade)
                    result.log(f'fde/test_{str(idx).zfill(2)}', agent_mean_fde)
                    result.log(f'mr/test_{str(idx).zfill(2)}', agent_mean_mr)

      


        if self.hparams.save_json:
            for i, p in enumerate(preds):
                write_dict = {}
                write_dict['preds'] = p.tolist() 
#                 write_dict['waypoint_preds'] = [waypoint_preds[0][i].tolist(), waypoint_preds[1][i].tolist()]
                write_dict['rotation'] = input_dict['ifc_helpers']['rotation'][i].tolist()
                write_dict['translation'] = input_dict['ifc_helpers']['translation'][i].tolist()
                write_dict['csv_file'] = input_dict['ifc_helpers']['csv_file'][i]
                write_dict['city'] = str(input_dict['ifc_helpers']['city'][i])
                
                
#                 result.log('preds', p.tolist())
# #                 result.log('waypoint_preds', agent_mean_ade)
#                 result.log('rotation', input_dict['ifc_helpers']['rotation'][i].tolist())
#                 result.log('translation', input_dict['ifc_helpers']['translation'][i].tolist())
#                 result.log('csv_file', input_dict['ifc_helpers']['csv_file'][i])
#                 result.log('city', str(input_dict['ifc_helpers']['city'][i]))
                
                write_dict['agent_labels'] = target_dict['agent_labels'][i].tolist()
                write_dict['agent_features'] = input_dict['agent_features'][i].tolist()
                write_dict['social_features'] = input_dict['social_features'][i].tolist()
                write_dict['social_label_features'] = input_dict['social_label_features'][i].tolist()
                if self.hparams.is_valid:
                    write_dict['att_weights'] = att_weights[i].tolist()
                    
                with open(self.hparams.save_dir + "/" + str(input_dict['ifc_helpers']['idx'][i]) + '.json', 'w') as json_file:
                    json.dump(write_dict, json_file, indent=4)
                                                                                      
        return result


    def test_epoch_end(self, outputs):
        result = pl.EvalResult()
        if self.hparams.is_valid:
            result_dict = {}
            for k in outputs.keys():
                if 'test' in k:
                    result.log(k, torch.mean(outputs[k]).item())
                    result_dict[k] = torch.mean(outputs[k]).item()
                    
#             print("save_json!!!!!!!!!!!!")
#             with open(self.hparams.save_dir + "/" + "test_results.json", 'w') as json_file:
#                 json.dump(self.json_data, json_file, indent=4)
        # return outputs
        return result
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.hparams.scheduler_step_size,
                                                         gamma=0.5)
        return [optimizer], [scheduler]

    def eval_preds(self, preds, target_dict, waypoint_preds=None):
        # Compute current k value
        k_value_threshold = self.hparams.k_value_threshold
        k_value_index = (self.current_epoch // k_value_threshold)
        if k_value_index >= len(self.hparams.k_values):
            k_value_index = len(self.hparams.k_values) - 1
        k_value = self.hparams.k_values[k_value_index]

        # Compute loss
        agent_preds = preds.narrow(dim=1, start=0, length=1).squeeze(1)
        if self.hparams.segment_CL or self.hparams.segment_CL_Prob or self.hparams.segment_CL_Gaussian_Prob:
            agent_waypoint_predictions = waypoint_preds.narrow(dim=1, start=0, length=1).squeeze(1)
        elif self.hparams.segment_CL_Encoder or self.hparams.segment_CL_Encoder_Gaussian or self.hparams.segment_CL_Encoder_Gaussian_Prob or self.hparams.segment_CL_Encoder_Prob:
            agent_waypoint_predictions = waypoint_preds[0].narrow(dim=1, start=0, length=1).squeeze(1)
            agent_encoder_waypoint_predictions = waypoint_preds[1].narrow(dim=1, start=0, length=1)
            social_encoder_waypoint_predictions = waypoint_preds[1].narrow(dim=1, start=1, length=waypoint_preds[1].size(1)-1)

        if agent_preds.size(-1) == 2:
            agent_loss = self.l1_ewta_loss(agent_preds, target_dict['agent_labels'], k=k_value)
        else:
            agent_loss = l1_ewta_loss_prob(agent_preds, target_dict['agent_labels'], k=k_value)
        if self.hparams.segment_CL or self.hparams.segment_CL_Prob or self.hparams.segment_CL_Gaussian_Prob:
            waypoint_loss = l1_ewta_waypoint_loss(agent_waypoint_predictions, target_dict['agent_labels'],
                                                  k_value, self.hparams.waypoint_step)
        elif self.hparams.segment_CL_Encoder or self.hparams.segment_CL_Encoder_Gaussian or \
                self.hparams.segment_CL_Encoder_Gaussian_Prob or self.hparams.segment_CL_Encoder_Prob:
            agent_waypoint_loss = l1_ewta_waypoint_loss(agent_waypoint_predictions, target_dict['agent_labels'],
                                                        k_value, self.hparams.waypoint_step)
            agent_encoder_waypoint_loss = l1_ewta_encoder_waypoint_loss(agent_encoder_waypoint_predictions,
                                                                        target_dict['agent_labels'],
                                                                        k_value)
            waypoint_loss = agent_waypoint_loss + agent_encoder_waypoint_loss
        else:
            waypoint_loss = 0.0
        total_loss = agent_loss + waypoint_loss

        # Compute metrics
        with torch.no_grad():
            if self.hparams.predict_delta:
                agent_preds_denorm = self.denorm_delta(agent_preds.detach(),
                                                       target_dict['agent_xy_ref_end'].unsqueeze(1))
                labels_preds_denorm = self.denorm_delta(target_dict['agent_labels'].detach(),
                                                        target_dict['agent_xy_ref_end'].unsqueeze(1))
                agent_mean_ade, agent_mean_fde, agent_mean_mr = compute_metrics(agent_preds_denorm, labels_preds_denorm)
            else:
                if agent_preds.size(-1) == 2:
                    if agent_preds.size(1) > 6:
                        agent_mean_ade, agent_mean_fde, agent_mean_mr = compute_metrics(
                            agent_preds.detach().narrow(1, 0, 6), target_dict['agent_labels'])
                    else:
                        agent_mean_ade, agent_mean_fde, agent_mean_mr = compute_metrics(
                            agent_preds.detach(), target_dict['agent_labels'])
                else:
                    if agent_preds.size(1) > 6:
                        probs = agent_preds.detach().narrow(-1, self.hparams.output_dim, 1).squeeze(-1)
                        probs_norm = nn.functional.softmax(torch.sum(probs, -1), -1)
                        _, sorted_indices = torch.sort(probs_norm, descending=True)
                        sorted_indices = sorted_indices.narrow(-1, 0, 6)
                        curr_preds = agent_preds.detach().narrow(-1, 0, self.hparams.output_dim).gather(
                            1, sorted_indices.unsqueeze(-1).unsqueeze(-1).expand(
                                -1, -1, agent_preds.size(2), self.hparams.output_dim))
                        agent_mean_ade, agent_mean_fde, agent_mean_mr = compute_metrics(
                            curr_preds, target_dict['agent_labels'])
                    else:
                        agent_mean_ade, agent_mean_fde, agent_mean_mr = compute_metrics(
                            agent_preds.detach().narrow(-1, 0, self.hparams.output_dim),
                            target_dict['agent_labels'])

        return total_loss, (agent_mean_ade, agent_mean_fde, agent_mean_mr)

# --
