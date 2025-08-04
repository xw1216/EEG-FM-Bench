import torch.nn as nn
import torch
from torch.autograd import Function


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class Conv2dWithConstraint(nn.Conv2d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv2dWithConstraint, self).forward(x)


class Conv1dWithConstraint(nn.Conv1d):
    '''
    Lawhern V J, Solon A J, Waytowich N R, et al. EEGNet: a compact convolutional neural network for EEG-based brain–computer interfaces[J]. Journal of neural engineering, 2018, 15(5): 056013.
    '''

    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv1dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(Conv1dWithConstraint, self).forward(x)


class LinearWithConstraint(nn.Linear):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(LinearWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        if self.doWeightNorm:
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        return super(LinearWithConstraint, self).forward(x)


def SMMDL_marginal(Cs, Ct):
    '''
    The SMMDL used in the CRGNet.
    Arg:
        Cs:The source input which shape is NxdXd.
        Ct:The target input which shape is Nxdxd.
    '''

    Cs = torch.mean(Cs, dim=0)
    Ct = torch.mean(Ct, dim=0)

    # loss = torch.mean((Cs-Ct)**2)
    loss = torch.mean(torch.mul((Cs - Ct), (Cs - Ct)))

    return loss


def SMMDL_conditional(Cs, s_label, Ct, t_label):
    '''
    The Conditional SMMDL of the source and target data.
    Arg:
        Cs:The source input which shape is NxdXd.
        s_label:The label of Cs data.
        Ct:The target input which shape is Nxdxd.
        t_label:The label of Ct data.
    '''
    s_label = s_label.reshape(-1)
    t_label = t_label.reshape(-1)

    class_unique = torch.unique(s_label)

    class_num = len(class_unique)
    all_loss = 0.0

    for c in class_unique:
        s_index = (s_label == c)
        t_index = (t_label == c)
        # print(t_index)
        if torch.sum(t_index) == 0:
            class_num -= 1
            continue
        c_Cs = Cs[s_index]
        c_Ct = Ct[t_index]
        m_Cs = torch.mean(c_Cs, dim=0)
        m_Ct = torch.mean(c_Ct, dim=0)
        loss = torch.mean((m_Cs - m_Ct) ** 2)
        all_loss += loss

    if class_num == 0:
        return 0

    return all_loss / class_num

