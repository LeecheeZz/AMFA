import torch.nn.functional as F
from torch.autograd import Variable
import torch
import torch.nn as nn

def cal_dkd_loss(outputs, outputs2, labels, loss_func):
    loss = torch.tensor(0, dtype=torch.float).cuda()
    for i in range(len(outputs)): # 6
        outputs_i = F.log_softmax(outputs[i], dim=1)
        outputs2_i = F.log_softmax(outputs2[i], dim=1)
        loss += loss_func(outputs_i, outputs2_i, labels)
        loss += loss_func(outputs2_i, outputs_i, labels) 
    loss = loss / (len(outputs) * 2)
    return loss
    
def dkd_loss(logits_student, logits_teacher, target, alpha, beta, temperature):
    gt_mask = _get_gt_mask(logits_student, target)
    other_mask = _get_other_mask(logits_student, target)
    pred_student = F.softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    pred_student = cat_mask(pred_student, gt_mask, other_mask)
    pred_teacher = cat_mask(pred_teacher, gt_mask, other_mask) # [8, 2]
    log_pred_student = torch.log(pred_student)
    tckd_loss = (
        F.kl_div(log_pred_student, pred_teacher,  reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    pred_teacher_part2 = F.softmax(
        logits_teacher / temperature - 1000.0 * gt_mask, dim=1
    )
    log_pred_student_part2 = F.log_softmax(
        logits_student / temperature - 1000.0 * gt_mask, dim=1
    )
    nckd_loss = (
        F.kl_div(log_pred_student_part2, pred_teacher_part2, reduction='sum')
        * (temperature**2)
        / target.shape[0]
    )
    return alpha * tckd_loss + beta * nckd_loss


def _get_gt_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask


def _get_other_mask(logits, target):
    target = target.reshape(-1)
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask


def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

class DKD(nn.Module):
    """Decoupled Knowledge Distillation(CVPR 2022)"""

    def __init__(self, alpha, beta, temperature):
        super(DKD, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.temperature = temperature

    def forward(self, logits_student, logits_teacher, target):
        # losses
        loss_dkd = dkd_loss(
            logits_student,
            logits_teacher,
            target,
            self.alpha,
            self.beta,
            self.temperature,
        )
        
        return loss_dkd