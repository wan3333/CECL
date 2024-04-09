import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.distributions import Categorical


class SupConLoss(nn.Module):
    """Following Supervised Contrastive Learning: 
        https://arxiv.org/pdf/2004.11362.pdf."""
    def __init__(self, temperature=7e-4, base_temperature=7e-2):
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(self, features, index=None, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if index is None:
            index = torch.ones(batch_size).bool()

        if mask is not None:
            # SupCon loss
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size][index], features.T),
                self.temperature)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )[index]
            mask = mask[index] * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)
        
            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
            loss = loss.mean()
        else:

            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size][index]
            k = features[batch_size:batch_size*2][index]
            queue = features[batch_size*2:]
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q, queue])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos, l_neg], dim=1)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits, labels)

        return loss


class CELoss(nn.Module):
    def forward(self, logits, labels, reduction='mean'):
        """
           :param logits: shape: (N, C)
           :param labels: shape: (N, C)
           :param reduction: options: "none", "mean", "sum"
           :return: loss or losses
           """
        N, C = logits.size
        assert labels.size(0) == N and labels.size(
            1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

        log_logits = F.log_softmax(logits, dim=1)
        losses = -torch.sum(log_logits * labels, dim=1)  # (N)

        if reduction == 'none':
            return losses
        elif reduction == 'mean':
            return torch.sum(losses) / logits.size(0)
        elif reduction == 'sum':
            return torch.sum(losses)
        else:
            raise AssertionError('reduction has to be none, mean or sum')


class EntropyLoss(nn.Module):
    def forward(self, logits):
        me = Categorical(probs=torch.softmax(input=logits, dim=1)).entropy().mean()
        return me


class CE_Soft_Label(nn.Module):
    def __init__(self):
        super().__init__()
        # print('Calculating uniform targets...')
        # calculate confidence
        self.confidence = None
        self.gamma = 2.0
        self.alpha = 0.25
    def init_confidence(self, noisy_labels, num_class):
        noisy_labels = torch.Tensor(noisy_labels).long().cuda()
        self.confidence = F.one_hot(noisy_labels, num_class).float().clone().detach()

    def forward(self, outputs, targets=None):
        logsm_outputs = F.log_softmax(outputs, dim=1)
        final_outputs = logsm_outputs * targets.detach()
        loss_vec = - ((final_outputs).sum(dim=1))
        #p = torch.exp(-loss_vec)
        #loss_vec =  (1 - p) ** self.gamma * loss_vec
        average_loss = loss_vec.mean()
        return loss_vec

