import torch
from torch import nn
import torch_sparse
from torch_geometric.utils import softmax
from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.nn import LayerNorm

# Define the ODE function.
# Input:
# --- t: A tensor with shape [], meaning the current time.
# --- x: A tensor with shape [#batches, dims], meaning the value of x at t.
# Output:
# --- dx/dt: A tensor with shape [#batches, dims], meaning the derivative of x at t.
class DiffusionAggregationODEFunc(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(DiffusionAggregationODEFunc, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.w = nn.Parameter(torch.eye(opt['hidden_dim']))
    self.d = nn.Parameter(torch.zeros(opt['hidden_dim']) + 1)
    self.alpha_sc = nn.Parameter(torch.ones(1))
    self.beta_sc = nn.Parameter(torch.ones(1))
    self.getW = SpGraphAttentionLayer(in_features, out_features, opt,
                                                     device).to(device) # W [E, 1]
    

    self.layernorm = LayerNorm(in_features)
    self.epoch = 0

  def sparse_multiply(self, x):
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    elif self.opt['block'] in ['mixed', 'hard_attention']:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.attention_weights, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.

    # x = self.layernorm(x)

    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    self.nfe += 1
    ax = self.sparse_multiply(x)
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
      beta = torch.sigmoid(self.beta_train)
    else:
      alpha = self.alpha_train
      beta = self.beta_train

    w = self.getW(x, self.edge_index) # W [E, 1]
    print(f'After calculate_gat_kernel: {torch.cuda.memory_allocated() / 1024 ** 2} MB')

    wx = torch.mean(torch.stack(
      [torch_sparse.spmm(self.edge_index, w[:, idx], x.shape[0], x.shape[0], x) for idx in
       range(w.shape[1])], dim = 0),
      dim = 0)

    awx = self.sparse_multiply(wx)

    f = alpha * (ax - x) + 4e-3 * beta * ((awx-ax) * x)
    print(f'End calculate_gat_kernel: {torch.cuda.memory_allocated() / 1024 ** 2} MB')
    # print(f.max())
    if self.opt['add_source']:
      f = f + self.beta_train * self.x0
    return f



class SpGraphAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.alpha = opt['leaky_relu_slope']
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = 1

    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features

    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']

    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    nn.init.xavier_normal_(self.W.data, gain=1.414)

    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)

    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414)

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge):
    wx = torch.mm(x, self.W)  # h: N x out
    h = wx.view(-1, self.h, self.d_k)
    h = h.transpose(1, 2)
    # Self-attention on the nodes - Shared attention mechanism
    # import pdb;pdb.set_trace()
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
     self.device)  # edge: 2*D x E
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)
    # edge_e = torch.log(torch.norm(h[edge[0, :], :, 0] - h[edge[1, :], :, 0], dim = 1)[...,None])
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention

  def __repr__(self):
    return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'