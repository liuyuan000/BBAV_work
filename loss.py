import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math


class BCELoss(nn.Module):
    def __init__(self):
        super(BCELoss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 1, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 1])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 1])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.binary_cross_entropy(pred.masked_select(mask),
                                          target.masked_select(mask),
                                          reduction='mean')
            return loss
        else:
            return 0.

class OffSmoothL1Loss(nn.Module):
    def __init__(self):
        super(OffSmoothL1Loss, self).__init__()

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _tranpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def forward(self, output, mask, ind, target):
        # torch.Size([1, 2, 152, 152])
        # torch.Size([1, 500])
        # torch.Size([1, 500])
        # torch.Size([1, 500, 2])
        pred = self._tranpose_and_gather_feat(output, ind)  # torch.Size([1, 500, 2])
        if mask.sum():
            mask = mask.unsqueeze(2).expand_as(pred).bool()
            loss = F.smooth_l1_loss(pred.masked_select(mask),
                                    target.masked_select(mask),
                                    reduction='mean')
            return loss
        else:
            return 0.

class FocalLoss(nn.Module):
  def __init__(self):
    super(FocalLoss, self).__init__()

  def forward(self, pred, gt):
      pos_inds = gt.eq(1).float()
      neg_inds = gt.lt(1).float()

      neg_weights = torch.pow(1 - gt, 4)

      loss = 0

      pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds
      neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

      num_pos  = pos_inds.float().sum()
      pos_loss = pos_loss.sum()
      neg_loss = neg_loss.sum()

      if num_pos == 0:
        loss = loss - neg_loss
      else:
        loss = loss - (pos_loss + neg_loss) / num_pos
      return loss

def isnan(x):
    return x != x

  
class LossAll(torch.nn.Module):
    def __init__(self):
        super(LossAll, self).__init__()
        self.L_hm = FocalLoss()
        self.L_wh =  OffSmoothL1Loss()
        self.L_off = OffSmoothL1Loss()
        self.L_cls_theta = BCELoss()
        # self.L_angle = nn.L1Loss(reduction='sum')
        # self.L_angle = nn.SmoothL1Loss(reduction='sum')
        self.L_angle = nn.MSELoss(reduction='sum')
        # self.L_angle = FocalLoss()
        self.print_sum = 0

    def forward(self, pr_decs, gt_batch):
        hm_loss  = self.L_hm(pr_decs['hm'], gt_batch['hm'])
        wh_loss  = self.L_wh(pr_decs['wh'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['wh'])
        off_loss = self.L_off(pr_decs['reg'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['reg'])
        ## add
        cls_theta_loss = self.L_cls_theta(pr_decs['cls_theta'], gt_batch['reg_mask'], gt_batch['ind'], gt_batch['cls_theta'])
        # angle_label = torch.ones_like(gt_batch['angle'])*2


        # pr_angle = pr_decs['angle']*math.pi*gt_batch['angle_mask']
        if self.print_sum % 10 == 0:
            print('hm loss is {}'.format(hm_loss),'wh loss is {}'.format(wh_loss), 'off loss is {}'.format(off_loss))
        self.print_sum += 1
        if self.print_sum > 1000:
            self.print_sum = 0
        '''
        pr_mask = pr_decs['hm'].ge(0.3)
        pr_angle = pr_mask*pr_decs['angle']*math.pi
        gt_angle = gt_batch['angle']*math.pi
        
        pr_circle_pos = []
        gt_circle_pos = []

        pr_circle_pos.append(torch.cos(pr_angle))
        pr_circle_pos.append(torch.sin(pr_angle))

        gt_circle_pos.append(torch.cos(gt_angle))
        gt_circle_pos.append(torch.sin(gt_angle))

        pr_circle_pos = torch.stack(pr_circle_pos, 1)
        gt_circle_pos = torch.stack(gt_circle_pos, 1)

        # angle_loss = self.L_angle(pr_circle_pos, gt_circle_pos)/(torch.sum(gt_batch['angle_mask'])+1)
        angle_loss = self.L_angle(pr_circle_pos, gt_circle_pos)/(torch.sum(pr_mask)+2)
        # angle_loss = angle_loss +  torch.mean(torch.abs(pr_decs['angle']))*0.1


        # angle_dev = torch.min(torch.abs(pr_angle-gt_angle), angle_label - torch.abs(pr_angle)-torch.abs(gt_angle))*gt_batch['angle_mask']
        # angle_ratio = Variable(torch.sum(pr_decs['hm'], dim=(1, ), keepdim=True))
        # angle_ratio = torch.sum(pr_decs['hm'], dim=(1, ), keepdim=True)
        # angle_taget = torch.zeros_like(angle_dev, dtype=torch.float32)
        # angle_loss = self.L_angle(angle_dev, angle_taget)*(gt_batch['angle_mask'].shape[2])*(gt_batch['angle_mask'].shape[3])/(torch.sum(gt_batch['angle_mask'])+1)
        # angle_loss = angle_loss + torch.abs(torch.mean(pr_angle))*0.1
        # angle_abs = angle_label - torch.abs(pr_decs['wh'][:,10,:,:]) - torch.abs(gt_batch['angle'])
        # angle_taget = torch.zeros_like(angle_abs)
        # angle_loss = self.L_angle(angle_abs, angle_taget)
        # angle_loss = self.L_angle(pr_decs['wh'][:,10,:,:], gt_batch['angle'])

        # if isnan(hm_loss) or isnan(wh_loss) or isnan(off_loss):
        
        # print(hm_loss)
        # print(wh_loss)
        # print(off_loss)
        # print(cls_theta_loss)
        # print('-----------------')
        '''

        loss =  hm_loss + wh_loss + off_loss + cls_theta_loss
        return loss
