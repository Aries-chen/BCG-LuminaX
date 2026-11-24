import torch
import torch.nn as nn
import pdb

#from .focal_loss import FocalLoss

class EventDetectionLoss(nn.Module):
    def __init__(self, loss_weights, weight = [1.0,1.0]):
        super(EventDetectionLoss, self).__init__()
        self.head_ratio = loss_weights['head_loss']
        self.iou_ratio = loss_weights['iou_loss']
        self.sig = nn.Sigmoid()
        self.weight = weight
        
    def iou_loss(self, labels, predictions):
        label_mask_1 = (labels == 1).float()
        predicted_labels = predictions
        pred_mask_1 = (predicted_labels > 0).float()

        def iou(mask1, mask2):
            intersection = torch.sum(mask1 * mask2)
            union = torch.sum(mask1) + torch.sum(mask2) - intersection
            return intersection / (union + 1e-6)

        iou_1 = iou(label_mask_1, pred_mask_1)
        iou_loss = 1 - iou_1
        return iou_loss

        
    def forward(self, net_output, sample):
        
        #print(net_output.shape)
        label = sample["cls_labels"].to(net_output.device)
        #print(label.)
        pred = net_output.float().reshape(-1, net_output.shape[-1])
        #pdb.set_trace()
        #label = torch.where(label.sum(dim=-1) > label.shape[-1]//2, torch.ones_like(label[:,0]), torch.zeros_like(label[:,0])).unsqueeze(-1)
        if len(label.size()) == 3:
            label = label.squeeze()

        head_loss = - self.weight[0]  * label * nn.LogSigmoid()(net_output.float()) -  self.weight[1] * (1 - label) * nn.LogSigmoid()(-net_output.float())
        iou_loss_value = self.iou_loss(label.reshape(-1), pred.reshape(-1))

        losses = {
            'head_loss': torch.mean(head_loss, dim=[-1,-2]),
            'iou_loss': self.iou_ratio*iou_loss_value,
            'loss': self.head_ratio*torch.mean(head_loss, dim=[-1,-2]) + self.iou_ratio*iou_loss_value,
        }
        return losses