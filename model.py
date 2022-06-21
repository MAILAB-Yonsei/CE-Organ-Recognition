import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import copy
import numpy as np
from loguru import logger
from torchinfo import summary
from tqdm import tqdm
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix
from eval import edit_score, f_score
import natsort
import matplotlib
import pandas as pd
from sklearn import metrics


def pos_encoding(dim, maxlen = 1000):
  pe = torch.Tensor(maxlen, dim)
  pos = torch.arange(0,maxlen, 1.).unsqueeze(1)
  k = torch.exp(-np.log(10000) * torch.arange(0,dim,2.)/dim)
  pe[:,0::2] = torch.sin(pos*k)
  pe[:,1::2] = torch.cos(pos*k)
  return pe

def softmax_with_temperature(z, T,dims=1) : 
    z = np.array(z)
    z = z / T 
    max_z = np.max(z,axis=dims) 
    exp_z = np.exp(z-max_z) 
    sum_exp_z = np.sum(exp_z,axis=dims)
    y = exp_z / sum_exp_z
    return y



class MS_TCN2(nn.Module):
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes):
        super(MS_TCN2, self).__init__()
        self.PG = Prediction_Generation(num_layers_PG, num_f_maps, dim, num_classes)
        self.Rs = nn.ModuleList([copy.deepcopy(Refinement(num_layers_R, num_f_maps, num_classes, num_classes)) for s in range(num_R)])

    def forward(self, x):
        out = self.PG(x)
        outputs = out.unsqueeze(0)
        for R in self.Rs:
            out = R(F.softmax(out, dim=1))
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)

        return outputs

class Prediction_Generation(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Prediction_Generation, self).__init__()

        self.num_layers = num_layers

        self.conv_1x1_in = nn.Conv1d(dim, num_f_maps, 1)

        self.conv_dilated_1 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**(num_layers-1-i), dilation=2**(num_layers-1-i))
            for i in range(num_layers)
        ))

        self.conv_dilated_2 = nn.ModuleList((
            nn.Conv1d(num_f_maps, num_f_maps, 3, padding=2**i, dilation=2**i)
            for i in range(num_layers)
        ))

        self.conv_fusion = nn.ModuleList((
             nn.Conv1d(2*num_f_maps, num_f_maps, 1)
             for i in range(num_layers)

            ))


        self.dropout = nn.Dropout()
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        f = self.conv_1x1_in(x) # input shape : (2,1024,10), output shape : (2,64,10)
        for i in range(self.num_layers):
            f_in = f
            f = self.conv_fusion[i](torch.cat([self.conv_dilated_1[i](f), self.conv_dilated_2[i](f)], 1))
            f = F.relu(f)
            f = self.dropout(f)
            f = f + f_in

        out = self.conv_out(f)
        return out

class Refinement(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(Refinement, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2**i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out)
        out = self.conv_out(out)
        return out
    
class MS_TCN(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MS_TCN, self).__init__()
        self.stage1 = SS_TCN(num_layers, num_f_maps, dim, num_classes)
        self.stages = nn.ModuleList([copy.deepcopy(SS_TCN(num_layers, num_f_maps, num_classes, num_classes)) for s in range(num_stages-1)])

    def forward(self, x, mask):
        out = self.stage1(x, mask)
        outputs = out.unsqueeze(0)
        for s in self.stages:
            out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
        return outputs


class SS_TCN(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SS_TCN, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList([copy.deepcopy(DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps)) for i in range(num_layers)])
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            # out = layer(out, mask)
            out = layer(out)
        out = self.conv_out(out) * mask[:, 0:1, :]
        return out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels):
        super(DilatedResidualLayer, self).__init__()
        self.conv_dilated = nn.Conv1d(in_channels, out_channels, 3, padding=dilation, dilation=dilation)
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return x + out


class Trainer:
    def __init__(self, num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes, dataset, split, num_stage, num_layer):
        # self.model = MS_TCN2(num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes) #num_layers_PG, num_layers_R, num_R, num_f_maps, dim, num_classes
        self.model = MS_TCN(num_stage, num_layer, num_f_maps, dim, num_classes) #num_stages, num_layers, num_f_maps, dim, num_classes
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

        logger.add('logs/' + dataset + "_" + split + "_{time}.log")
        logger.add(sys.stdout, colorize=True, format="{message}")

    def train(self, save_dir, data_loader, data_loader_val, data_loader_test, num_epochs, batch_size, learning_rate, device):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.98)
        val_loss = 1000.0
        
        overlap = [.5, .75, .9]
        best_acc = 0
        for epoch in range(num_epochs):
            epoch_loss = 0
            last_loss = 0
            correct = 0
            total = 0
            targets = []
            predicts = []
            edit = 0
            tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
            temp = torch.empty(3, 3)
            cm = torch.zeros_like(temp)
            del temp
            for i, data in enumerate(tqdm(data_loader)):
                self.model.train()
                batch_input, batch_target = data
                batch_input = torch.Tensor(batch_input.float())
                
                batch_target = torch.Tensor(batch_target.float())
                
                randnum = torch.randn(1,)
                
                
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                

                
                batch_input = batch_input.permute((0,2,1))
                predictions = self.model(batch_input)
                # mask = torch.ones_like(batch_input).to(device)
                # predictions = self.model(batch_input,mask) #for ms-tcn

                loss = 0
                for p in predictions:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), (batch_target.long()).view(-1)) 
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)) 

                
                epoch_loss += loss.item()
                last_loss += (self.ce(predictions[-1].transpose(2, 1).contiguous().view(-1, self.num_classes), (batch_target.long()).view(-1)) + 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(predictions[-1][:, :, 1:], dim=1), F.log_softmax(predictions[-1].detach()[:, :, :-1], dim=1)), min=0, max=16))).item()
                (loss/4).backward()
                if (i+1) % 4 == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                
                _, predicted = torch.max(predictions[-1].data, 1)
            
                
                correct += ((predicted == batch_target).float().squeeze(1)).sum().item()
                total += batch_target.shape[0]*batch_target.shape[-1]
                
                targets.extend(batch_target[0].cpu().detach().numpy())
                predicts.extend(predicted[0].cpu().detach().numpy())
                
                edit += edit_score(predicted[0].cpu().detach().numpy(), batch_target[0].cpu().detach().numpy())
                
                
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted[0].cpu().detach().numpy(), batch_target[0].cpu().detach().numpy(), overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
            
            targets = np.array(targets)
            predicts = np.array(predicts)
            
            cm = confusion_matrix(targets,predicts)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            precision = np.diag(cm) / np.sum(cm, axis=0)
            # recall = recall[1:]
            # precision = precision[1:]
            
            scheduler.step()
            logger.info("[epoch %d]: epoch loss = %f, last loss = %f, acc = %f, recall = %f, precision = %f" % (epoch + 1, epoch_loss / len(data_loader), last_loss /len(data_loader),float(correct)/total,np.mean(recall), np.mean(precision)))
            
            print('Edit: %.4f' % ((1.0*edit)/len(data_loader)))
            
            for s in range(len(overlap)):
                precision = tp[s] / float(tp[s]+fp[s])
                recall = tp[s] / float(tp[s]+fn[s])

                f1 = 2.0 * (precision*recall) / (precision+recall)

                f1 = np.nan_to_num(f1)*100
                print('F1@%0.2f: %.4f' % (overlap[s], f1))
            
            targets = []
            predicts = []
            tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
            edit = 0
            epoch_val_loss = 0
            correct_v = 0
            total_v = 0
            last_loss = 0
            
            for i, data in enumerate(tqdm(data_loader_val)):
                self.model.eval()
                batch_input_v, batch_target_v = data
                batch_input_v = torch.Tensor(batch_input_v.float())
                batch_target_v = torch.Tensor(batch_target_v.float())
                
                batch_input_v = batch_input_v.to(device)
                batch_target_v = batch_target_v.to(device)
                
                
                batch_input_v = batch_input_v.permute((0,2,1))
                predictions_v = self.model(batch_input_v)
                # mask = torch.ones_like(batch_input_v).to(device)
                # predictions_v = self.model(batch_input_v,mask)#for ms-tcn

                loss = 0
                for p in predictions_v:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), (batch_target_v.long()).view(-1)) 
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)) 
                
                last_loss += (self.ce(predictions_v[-1].transpose(2, 1).contiguous().view(-1, self.num_classes), (batch_target_v.long()).view(-1)) + 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(predictions_v[-1][:, :, 1:], dim=1), F.log_softmax(predictions_v[-1].detach()[:, :, :-1], dim=1)), min=0, max=16))).item()

                epoch_val_loss += loss
                _, predicted = torch.max(predictions_v[-1].data, 1)
                
                
                correct_v += ((predicted == batch_target_v).float().squeeze(1)).sum().item()
                total_v += batch_target_v.shape[0]*batch_target_v.shape[-1]
                
                targets.extend(batch_target_v[0].cpu().detach().numpy())
                predicts.extend(predicted[0].cpu().detach().numpy())
                
                edit += edit_score(predicted[0].cpu().detach().numpy(), batch_target_v[0].cpu().detach().numpy())
                
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted[0].cpu().detach().numpy(), batch_target_v[0].cpu().detach().numpy(), overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
            
            targets = np.array(targets)
            predicts = np.array(predicts)
            
            cm = confusion_matrix(targets,predicts)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            precision = np.diag(cm) / np.sum(cm, axis=0)
        
            recall = recall[1:]
            precision = precision[1:]
            
            if float(correct_v)/total_v > best_acc:
                best_acc = float(correct_v)/total_v
                torch.save(self.model.state_dict(), save_dir + "/best_epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/best_epoch-" + str(epoch + 1) + ".opt")
            
            logger.info("[epoch %d]: epoch val loss = %f, val last loss = %f,  val acc = %f, recall = %f, precision = %f" % (epoch + 1, epoch_val_loss / len(data_loader_val), last_loss / len(data_loader_val),
                                                               float(correct_v)/total_v, np.mean(recall), np.mean(precision)))
            
            print('val Edit: %.4f' % ((1.0*edit)/len(data_loader_val)))
            for s in range(len(overlap)):
                precision = tp[s] / float(tp[s]+fp[s])
                recall = tp[s] / float(tp[s]+fn[s])

                f1 = 2.0 * (precision*recall) / (precision+recall)

                f1 = np.nan_to_num(f1)*100
                print('val F1@%0.2f: %.4f' % (overlap[s], f1))
                
            if val_loss > last_loss:
                val_loss = last_loss
                torch.save(self.model.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".model")
                torch.save(optimizer.state_dict(), save_dir + "/epoch-" + str(epoch + 1) + ".opt")
            
            targets = []
            predicts = []
            tp, fp, fn = np.zeros(3), np.zeros(3), np.zeros(3)
            epoch_val_loss = 0
            correct_v = 0
            total_v = 0
            edit = 0
            last_loss = 0
            
            for i, data in enumerate(tqdm(data_loader_test)):
                self.model.eval()
                batch_input_v, batch_target_v = data
                batch_input_v = torch.Tensor(batch_input_v.float())
                batch_target_v = torch.Tensor(batch_target_v.float())
                
                batch_input_v = batch_input_v.to(device)
                batch_target_v = batch_target_v.to(device)
                
                
                batch_input_v = batch_input_v.permute((0,2,1))
                mask = torch.ones_like(batch_input_v).to(device)
                predictions_v = self.model(batch_input_v)
                # predictions_v = self.model(batch_input_v,mask) #for ms-tcn
                loss = 0
                for p in predictions_v:
                    loss += self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_classes), (batch_target_v.long()).view(-1)) 
                    loss += 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0, max=16)) 
                
                last_loss += (self.ce(predictions_v[-1].transpose(2, 1).contiguous().view(-1, self.num_classes), (batch_target_v.long()).view(-1)) + 0.15*torch.mean(torch.clamp(self.mse(F.log_softmax(predictions_v[-1][:, :, 1:], dim=1), F.log_softmax(predictions_v[-1].detach()[:, :, :-1], dim=1)), min=0, max=16))).item()

                epoch_val_loss += loss
                _, predicted = torch.max(predictions_v[-1].data, 1)
                
                correct_v += ((predicted == batch_target_v).float().squeeze(1)).sum().item()
                total_v += batch_target_v.shape[0]*batch_target_v.shape[-1]
                
                targets.extend(batch_target_v[0].cpu().detach().numpy())
                predicts.extend(predicted[0].cpu().detach().numpy())
                
                edit += edit_score(predicted[0].cpu().detach().numpy(), batch_target_v[0].cpu().detach().numpy())
                
                for s in range(len(overlap)):
                    tp1, fp1, fn1 = f_score(predicted[0].cpu().detach().numpy(), batch_target_v[0].cpu().detach().numpy(), overlap[s])
                    tp[s] += tp1
                    fp[s] += fp1
                    fn[s] += fn1
            
            targets = np.array(targets)
            predicts = np.array(predicts)
            
            cm = confusion_matrix(targets,predicts)
            recall = np.diag(cm) / np.sum(cm, axis=1)
            precision = np.diag(cm) / np.sum(cm, axis=0)
            recall = recall[1:]
            precision = precision[1:]
            
            logger.info("[epoch %d]: epoch test loss = %f,  test last loss = %f, test acc = %f, recall = %f, precision = %f" % (epoch + 1, epoch_val_loss / len(data_loader_test), last_loss / len(data_loader_test),
                                                               float(correct_v)/total_v, np.mean(recall), np.mean(precision)))

            print('test Edit: %.4f' % ((1.0*edit)/len(data_loader_test)))
            for s in range(len(overlap)):
                precision = tp[s] / float(tp[s]+fp[s])
                recall = tp[s] / float(tp[s]+fn[s])

                f1 = 2.0 * (precision*recall) / (precision+recall)

                f1 = np.nan_to_num(f1)*100
                print('test F1@%0.2f: %.4f' % (overlap[s], f1))            
    
                    
                
   

    