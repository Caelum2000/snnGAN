import abc
import math
from abc import ABC, abstractmethod
from select import select
from turtle import forward
from typing import ForwardRef

import numpy as np
import torch
from torch import nn
from torch.nn import Parameter
from torch.nn import functional as F

ALPHA = 2.


class AtanGrad(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, alpha):
        ctx.save_for_backward(inputs, alpha)
        return inputs.gt(0.).float()

    @staticmethod
    def backward(ctx, grad_output):
        grad_x = None
        grad_alpha = None

        # saved_tensor[0] == nan here !
        shared_c = grad_output / (1 + (ctx.saved_tensors[1] * math.pi / 2 *
                                       ctx.saved_tensors[0]).square())
        if ctx.needs_input_grad[0]:
            grad_x = ctx.saved_tensors[1] / 2 * shared_c
        if ctx.needs_input_grad[1]:
            # 由于alpha只有一个元素，因此梯度需要求和，变成标量
            grad_alpha = (ctx.saved_tensors[0] / 2 * shared_c).sum()
        '''with torch.no_grad():
            print(grad_output.mean())
            print(ctx.saved_tensors[0].mean())
            print(ctx.saved_tensors[1].mean())
            print(" ")'''

        return grad_x, grad_alpha


class BaseNode(nn.Module, abc.ABC):

    def __init__(self, threshold=1., weight_warmup=False, V_reset=0.):
        super(BaseNode, self).__init__()
        self.threshold = threshold
        self.mem = 0.
        self.spike = 0.
        self.weight_warmup = weight_warmup  # 一般在训练静态数据集 较深网络时使用

    @abc.abstractmethod
    def calc_spike(self):
        pass

    @abc.abstractmethod
    def integral(self, inputs):
        pass

    def forward(self, inputs):
        if self.weight_warmup:
            return inputs
        else:
            self.integral(inputs)
            self.calc_spike()
            return self.spike

    def n_reset(self):
        self.mem = 0.
        self.spike = 0.

    def get_n_fire_rate(self):
        if self.spike is None:
            return 0.
        return float((self.spike.detach() >= self.threshold).sum()) / float(
            np.product(self.spike.shape))


class DotProductAttention(nn.Module):

    def __init__(self, **kwargs):
        super(DotProductAttention, self).__init__(**kwargs)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys):
        # note that: q_size == k_size
        # queries.shape = (b,n_q,q_size)
        # keys.shape = (b,n_k,k_size)
        # values.shape = (b,n_k,v_size)
        d = queries.shape[-1]
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(
            d)  # (b,q,qs)*(b,ks,k) -> (b,q,k)
        self.attention_weights = torch.softmax(scores, dim=-1)
        return self.attention_weights


class AttentionScoring_1(nn.Module):

    def __init__(self, query_size, key_size, num_hiddens):
        super().__init__()
        self.W_q = nn.Linear(query_size, num_hiddens)
        self.W_k = nn.Linear(key_size, num_hiddens)
        self.attention = DotProductAttention()

    def forward(self, queries, keys):
        # queries.shape = (b,2,784)
        # keys.shape = (b,1,784)
        queries = self.W_q(queries)
        keys = self.W_k(keys)
        self.attention_weights = self.attention(queries, keys)
        return self.attention_weights


class HasInfoNCELoss(ABC):
    r"""
    This class is for those modules that need to compute infoNCE loss
    """

    def __init__(self) -> None:
        super().__init__()
        self._project_x = None
        self._project_latent = None

    def batch_infonce_loss(self, X, X_latent):
        r"""
        This function generates positive/negative examples by same/different examples for a minibatch 
        X.shape = (b,...,original_dim)
        X_latent.shape = (b,...,latent_dim)
        """
        # print(X.device)
        # print(next(self._project_x.parameters()).device)
        projected_X = self._project_x(X).unsqueeze(-2).unsqueeze(
            0)  # (b,...,origianl_dim) => (1,b,...,1,compare_vec_dim)
        projected_latent = self._project_latent(X_latent).unsqueeze(
            -1).unsqueeze(
                1)  # (b,...,latent_dim) => (b,1,...,compare_vec_dim,1)

        score_map = torch.matmul(projected_X,
                                 projected_latent).squeeze(-1).squeeze(
                                     -1)  # (bl,bm,...)

        batch_size = score_map.shape[0]
        labels = torch.ones(
            score_map.shape[1:], device=X.device) * torch.arange(
                batch_size, device=X.device).reshape(
                    [batch_size] + [1] * (len(score_map.shape) - 2))  # (b,...)
        # print(labels.device)
        l = F.cross_entropy(score_map, labels.long(),
                            reduction='mean')  # scalar
        return l

    '''@property
    @abstractmethod
    def _project_x(self):
        pass

    @property
    @abstractmethod
    def _project_latent(self):
        pass'''

    @abstractmethod
    def compute_infonce_loss(self):
        pass


class AttentionScoring_2(nn.Module, HasInfoNCELoss):
    r"""
    utilizing infoNCE method
    """

    def __init__(self, query_size, key_size, num_hiddens, latent_dim,
                 compare_dim):
        super().__init__()

        # mapping for queries
        self.mq = nn.Sequential(nn.Linear(query_size, num_hiddens), nn.ReLU(),
                                nn.Linear(num_hiddens, latent_dim))

        # mapping for keys
        self.mk = nn.Sequential(nn.Linear(key_size, num_hiddens), nn.ReLU(),
                                nn.Linear(num_hiddens, latent_dim))

        self.attention = DotProductAttention()
        self.compare_dim = compare_dim
        self.query_size = query_size
        self.key_size = key_size
        self.latent_dim = latent_dim
        self._project_x = nn.Linear(self.query_size, self.compare_dim)
        self._project_latent = nn.Linear(self.latent_dim, self.compare_dim)

    def forward(self, queries, keys):
        # queries.shape = (b,2,784)
        # keys.shape = (b,1,784)
        queries_latent = self.mq(queries)  # (b,2,latent_dim)
        keys_latent = self.mk(keys)  # (b,1,latent_dim)
        self.attention_weights = self.attention(queries_latent, keys_latent)
        infonce_loss = self.compute_infonce_loss(queries, queries_latent, keys,
                                                 keys_latent)
        return self.attention_weights, infonce_loss

    '''@property
    def _project_x(self):
        return nn.Linear(self.query_size, self.compare_dim)

    @property
    def _project_latent(self):
        return nn.Linear(self.latent_dim, self.compare_dim)'''

    def compute_infonce_loss(self, queries, queries_latent, keys, keys_latent):
        return (self.batch_infonce_loss(queries, queries_latent) +
                self.batch_infonce_loss(keys, keys_latent)) / 2


class LIFNode(BaseNode):

    def __init__(self, threshold=1., tau=2., act_fun=AtanGrad):
        super().__init__(threshold)
        self.tau = tau
        self.act_fun = act_fun.apply

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.act_fun(self.mem - self.threshold, torch.tensor(2.))
        self.mem = self.mem * (1 - self.spike.detach())


class MPNode(BaseNode):

    def __init__(self, tau=2.0):
        super().__init__()
        self.tau = tau

    def integral(self, inputs):
        self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        # self.spike = self.act_fun(self.mem - self.threshold, torch.tensor(2.))
        # self.mem = self.mem * (1 - self.spike.detach())
        self.spike = self.mem


class ScoringMP(BaseNode):

    def __init__(self, scoring_mode, tau=2.0):
        super().__init__()
        self.tau = tau
        self.scoring_mode = scoring_mode
        self.scoring_function = None
        if self.scoring_mode == 'ScoringNet_1':
            self.scoring_function = ScoringNet_1(num_inputs=784,
                                                 num_hiddens=256)
        elif self.scoring_mode == 'AttentionScoring_1':
            self.scoring_function = AttentionScoring_1(784, 784, 256)
        elif self.scoring_mode == 'AttentionScoring_2':
            self.scoring_function = AttentionScoring_2(784, 784, 256, 128, 64)

    def integral(self, inputs):
        # inputs.shape = (b,1,28,28) or (b,784)
        if isinstance(self.mem, float):
            # print("Hello")
            self.mem = inputs
        else:
            if self.scoring_mode == 'ScoringNet_1':
                batch_size = inputs.shape[0]
                mem_score = self.scoring_function(
                    self.mem.reshape((batch_size, -1)))  # (b,1)
                inputs_score = self.scoring_function(
                    inputs.reshape((batch_size, -1)))  # (b,1)
                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)
            elif self.scoring_mode == 'AttentionScoring_1':
                batch_size = inputs.shape[0]
                if len(inputs.shape) == 4:
                    inputs_temp = inputs.reshape((batch_size, -1))  # (b,784)
                    mem_temp = self.mem.reshape((batch_size, -1))  # (b,784)
                else:
                    inputs_temp = inputs
                    mem_temp = self.mem

                # (b,784) => (b,1,784)
                inputs_temp = inputs_temp.unsqueeze(1)
                mem_temp = mem_temp.unsqueeze(1)

                keys = torch.cat([mem_temp, inputs_temp], dim=1)  # (b,2,784)
                attention_weights = self.scoring_function(mem_temp,
                                                          keys)  # (b,1,2)
                mem_score = attention_weights[:, 0, 0]  # (b,)
                inputs_score = attention_weights[:, 0, 1]  # (b,)

                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)

                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)
            elif self.scoring_mode == "AttentionScoring_2":
                # print("hello")
                batch_size = inputs.shape[0]
                if len(inputs.shape) == 4:
                    inputs_temp = inputs.reshape((batch_size, -1))  # (b,784)
                    mem_temp = self.mem.reshape((batch_size, -1))  # (b,784)
                else:
                    inputs_temp = inputs
                    mem_temp = self.mem

                # (b,784) => (b,1,784)
                inputs_temp = inputs_temp.unsqueeze(1)
                mem_temp = mem_temp.unsqueeze(1)

                keys = torch.cat([mem_temp, inputs_temp], dim=1)  # (b,2,784)
                attention_weights, infonce_loss = self.scoring_function(
                    mem_temp, keys)  # (b,1,2)
                mem_score = attention_weights[:, 0, 0]  # (b,)
                inputs_score = attention_weights[:, 0, 1]  # (b,)

                mem_score = mem_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)
                inputs_score = inputs_score.reshape(
                    [batch_size, 1] + [1] *
                    (len(inputs.shape) - 2))  # (b,1,1,1)

                self.mem = (mem_score * self.mem +
                            inputs_score * inputs) / (mem_score + inputs_score)

                # print(infonce_loss)

                return infonce_loss

            elif self.scoring_mode is None:
                self.mem = self.mem + (inputs - self.mem) / self.tau

    def calc_spike(self):
        self.spike = self.mem

    def forward(self, inputs):
        if self.weight_warmup:
            return inputs
        elif self.scoring_mode == "AttentionScoring_2":
            infonce_loss = self.integral(inputs)
            self.calc_spike()
            return self.spike, infonce_loss
        else:
            self.integral(inputs)
            self.calc_spike()
            return self.spike
