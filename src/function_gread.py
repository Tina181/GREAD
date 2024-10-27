import torch
from torch import nn
import torch_sparse
from torch.nn.init import uniform, xavier_uniform_
from base_classes import ODEFunc
from utils import MaxNFEException
from torch_geometric.utils import softmax
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import degree 
import torch.nn.functional as F
import math
from torch.nn import LayerNorm


"""
Define the ODE function.
Input:
 - t: A tensor with shape [], meaning the current time.
 - x: A tensor with shape [#nodes, dims], meaning the value of x at t.
Output:
 - dx/dt: A tensor with shape [#nodes, dims], meaning the derivative of x at t.
"""
class ODEFuncGread(ODEFunc):

  # currently requires in_features = out_features
  def __init__(self, in_features, out_features, opt, data, device):
    super(ODEFuncGread, self).__init__(opt, data, device)

    self.in_features = in_features
    self.out_features = out_features
    self.diffusion_rate1 = opt['diffusion_rate1']
    self.diffusion_rate2 = opt['diffusion_rate2']
    self.layer_norm = LayerNorm(out_features)
    self.nfe = 0
    if opt['reaction_term']=='aggdiff-gat':
      self.GAT_Kernel = SpGraphAttentionLayer(self.in_features, self.out_features, self.opt, self.device)
    if opt['reaction_term']=='aggdiff-sin':
      self.sin_w = nn.Parameter(torch.tensor(1.0), requires_grad=True)
      self.sin_b = nn.Parameter(torch.tensor(1.0), requires_grad=True)
      self.Sin_passing = SineMessagePassing(in_features=self.in_features, out_features=self.out_features)
    if opt['reaction_term']=='aggdiff-log':
      self.Log_Kernel = SpGraphlogKernelLayer(self.in_features, self.out_features, self.opt, self.device)
    if opt['reaction_term']=='aggdiff-gauss':
      self.Gauss_Kernel =SpGraphgaussKernelLayer(self.in_features, self.out_features, self.opt, self.device)
    
    self.reaction_tanh = False
    if opt['beta_diag'] == True:
      self.b_W = nn.Parameter(torch.Tensor(in_features))
      self.reset_parameters()
    self.epoch = 0
  
  def reset_parameters(self):
    if self.opt['beta_diag'] == True:
      uniform(self.b_W, a=-1, b=1)
  
  def set_Beta(self, T=None):
    Beta = torch.diag(self.b_W)
    return Beta

  def sparse_multiply(self, x):
    """
    - `attention` is equivalent to "Soft Adjacency Matrix (SA)".
    - If `block` is `constant`, we use "Original Adjacency Matrix (OA)"
    """
    if self.opt['block'] in ['attention']:  # adj is a multihead attention
      mean_attention = self.attention_weights.mean(dim=1)
      ax = torch_sparse.spmm(self.edge_index, mean_attention, x.shape[0], x.shape[0], x)
    else:  # adj is a torch sparse matrix
      ax = torch_sparse.spmm(self.edge_index, self.edge_weight, x.shape[0], x.shape[0], x)
    return ax

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:
      raise MaxNFEException
    if self.opt["layer_norm"]:
      x = self.layer_norm(x)
    self.nfe += 1
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train(t))
      beta = torch.sigmoid(self.beta_train(t))
    else:
      alpha = self.alpha_train(t) * self.diffusion_rate1
      beta = self.beta_train(t) * self.diffusion_rate2

    """
    - `x` is equivalent $H$ in our paper.
    - `diffusion` is the diffusion term.
    """
    ax = self.sparse_multiply(x)
    diffusion = (ax - x)

    """
    - `reaction` is the reaction term.
    - We consider four `reaction_term` options
     - When `reaction_term` is bspm: GREAD-BS
     - When `reaction_term` is fisher: GREAD-F
     - When `reaction_term` is allen-cahn: GREAD-AC
     - When `reaction_term` is zeldovich: GREAD-Z
    - The `tanh` on reaction variable is optional, but we don't use in our experiments.
    """
    if self.opt['reaction_term'] == 'bspm':
      reaction = -self.sparse_multiply(diffusion) # A(AX-X)
    elif self.opt['reaction_term'] == 'fisher':
      reaction = -(x-1)*x
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'allen-cahn':
      reaction = -(x**2-1)*x
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'zeldovich':
      reaction = -(x**2-x)*x
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] =='st':
      reaction = self.x0
    elif self.opt['reaction_term'] == 'fb':
      ax = -self.sparse_multiply(x)
      reaction = x - ax # L = I - A
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'fb3':
      ax = -self.sparse_multiply(x)
      reaction = x - ax# L = I - A
      if self.reaction_tanh == True:
        reaction = torch.tanh(reaction)
    elif self.opt['reaction_term'] == 'aggdiff-gat':
      k = self.GAT_Kernel(x, self.edge_index)  # torch.Size([10138, 1])
      k = k.to(self.device)
      kx = torch.zeros_like(x).to(self.device)
      for idx in range(k.shape[1]):
          kx += torch_sparse.spmm(self.edge_index, k[:, idx], x.shape[0], x.shape[0], x)
      kx /= k.shape[1]  # 取平均值
      kx = kx.to(self.device)
      reaction =  (ax-x)*kx
    elif self.opt['reaction_term']=='aggdiff-sin':
      out = self.Sin_passing(x, self.edge_index).to(self.device)  # torch.Size([2485, 64])
      reaction = self.sin_w+ self.sin_b*out  # 1e-4 is a hyperparameter to avoid gradient explosion
    elif self.opt['reaction_term'] =='aggdiff-log':
      n = x.shape[0]  # number of nodes
      k = self.Log_Kernel(x, self.edge_index)  # [10138]
      kx = torch_sparse.spmm(self.edge_index, k, n, n, x) # [2485, 64]
      kx = kx.to(self.device)
      reaction = (ax-x)*kx  # 1e-4 is a hyperparameter to avoid gradient explosion
    elif self.opt['reaction_term'] =='aggdiff-gauss':
      n = x.shape[0]  # number of nodes 2485
      k = self.Gauss_Kernel(x, self.edge_index)  # torch.Size([10138])
      kx = torch_sparse.spmm(self.edge_index, k, n, n, x) # [2485, 64]
      kx = kx.to(self.device)
      reaction = (ax-x)*kx  # 1e-4 is a hyperparameter to avoid gradient explosion
    else:
      raise Exception('Unknown reaction term.')
    
    """
    - `f` is in the reaction-diffusion form
     - $\mathbf{f}(\mathbf{H}(t)) := \frac{d \mathbf{H}(t)}{dt} = -\alpha\mathbf{L}\mathbf{H}(t) + \beta\mathbf{r}(\mathbf{H}(t), \mathbf{A})$
    - `beta_diag` is equivalent to $\beta$ with VC dimension
     - `self.Beta` is diagonal matrix intialized with gaussian distribution
     - Due to the diagonal matrix, it is same to the result of `beta*reaction` when `beta` is initialized with gaussian distribution.
    """
    if self.opt['beta_diag'] == False:
      if self.opt['reaction_term'] =='fb':
        f = alpha*diffusion + beta*reaction
      elif self.opt['reaction_term'] =='fb3':
        f = alpha*diffusion + beta*(reaction + x)
      else:
        f = alpha*diffusion + beta*reaction
    elif self.opt['beta_diag'] == True:
      f = alpha*diffusion + reaction@self.Beta
    
    """
    - We do not use the `add_source` on GREAD
    """
    if self.opt['add_source']:
      f = f + self.source_train(t) * self.x0
    return f




class SpGraphAttentionLayer(nn.Module):
  """
  Sparse version GAT layer, similar to https://arxiv.org/abs/1710.10903
  """

  def __init__(self, in_features, out_features, opt, device, concat=True):
    super(SpGraphAttentionLayer, self).__init__()
    self.in_features = in_features  # 64
    self.out_features = out_features  # 64
    self.alpha = opt['leaky_relu_slope']  # 0.2
    self.concat = concat
    self.device = device
    self.opt = opt
    self.h = 1
    try:
      self.attention_dim = opt['attention_dim']
    except KeyError:
      self.attention_dim = out_features
    # opt['heads']: 4 
    assert self.attention_dim % opt['heads'] == 0, "Number of heads must be a factor of the dimension size"
    self.d_k = self.attention_dim // opt['heads']
    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim)))
    nn.init.xavier_normal_(self.W.data, gain=1.414) # torch.Size([64, 64])
    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features)))
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)  # torch.Size([64, 64])
    # self.d_k:16 （每个头的维数）
    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1)))
    nn.init.xavier_normal_(self.a.data, gain=1.414) # torch.Size([32, 1, 1])

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge): # x: torch.Size([2485, 64]), edge: 2 x E  torch.Size([2, 10138])
    wx = torch.mm(x.to(self.device), self.W.to(self.device))  # h: N x out  wx: torch.Size([2485, 64])
    h = wx.view(-1, self.h, self.d_k) # torch.Size([2485, 1, 64])
    h = h.transpose(1, 2) # torch.Size([2485, 64, 1])
    h=h.to(self.device)
    # Self-attention on the nodes - Shared attention mechanism
    # import pdb;pdb.set_trace()
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
     self.device)  # edge: 2*D x E  torch.Size([128, 10138, 1])
    # self.a : torch.Size([128, 1, 1])
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)  # torch.Size([10138, 1])
    # edge_e = torch.log(torch.norm(h[edge[0, :], :, 0] - h[edge[1, :], :, 0], dim = 1)[...,None])
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention

class SineMessagePassing(MessagePassing):
    def __init__(self, in_features, out_features):
        super().__init__(aggr='mean')  # 使用'add'聚合
        # self.lin = torch.nn.Linear(in_features, out_features)
    def forward(self, x, edge_index):
        # x shape: [N, in_channels]
        # edge_index shape [2, E]
        # Step 1: 线性变换节点特征矩阵
        # x = self.lin(x) # x [2485,64]
        # Step 2: 计算规范化系数
        row, col = edge_index # [10138], [10138]
        deg = degree(col, x.size(0), dtype=x.dtype) # degree vector of each node (count the number of nodes appear in col) [2485] tensor([3., 3., 5.,  ..., 2., 4., 4.], device='cuda:0')
        deg_inv = 1 / deg # the inverse degree of each node tensor([0.3333, 0.3333, 0.2000,  ..., 0.5000, 0.2500, 0.2500], device='cuda:0')
        deg_inv[deg_inv == float('inf')] = 0  # avoid infs in case of isolated nodes 1/0=inf
        norm = deg_inv[row]  # 对于每条边的源节点进行归一化 1/d_i [10138]
        # Step 3: 计算正弦消息并聚合
        return self.propagate(edge_index, x=x, norm=norm)
    def message(self,x_i, x_j, norm):
        # x_j has shape [E, out_channels]
        # x_i: 源节点, x_j: 目标节点  torch.Size([10138, 64])
        # Step 4: 乘以归一化项
        # norm.view(-1, 1)变成列向量 torch.Size([10138, 1])
        return norm.view(-1, 1) * torch.sin(x_j-x_i)  # torch.Size([10138, 64])   
    def update(self, aggr_out):
        # 这里我们可以选择是否对输出进行进一步的处理
        return aggr_out # torch.Size([2485, 64])

class SpGraphlogKernelLayer(nn.Module):
  def __init__(self, in_features, out_features, opt, device):
    super(SpGraphlogKernelLayer, self).__init__()
    self.in_features = in_features  # 64
    self.out_features = out_features  # 64
    self.device = device
    self.opt = opt
    self.epsilon = opt['log_eps'] # avoid nan in log kernel  
  def forward(self, x, edge): # x: torch.Size([2485, 64]),
    
    k = torch.log(self.epsilon + torch.norm(x[edge[0, :], :] - x[edge[1, :], :], dim = 1)).to(self.device)
    return k  # torch.Size([10138])
  
class SpGraphgaussKernelLayer(nn.Module):
  def __init__(self, in_features, out_features, opt, device):
    super(SpGraphgaussKernelLayer, self).__init__()
    self.in_features = in_features  # 64
    self.out_features = out_features  # 64
    self.device = device
    self.opt = opt
    self.epsilon = 1  
  def forward(self, x, edge): # x: torch.Size([2485, 64]),
    d = x.shape[1]
    sq_dist = torch.norm(x[edge[0, :], :] - x[edge[1, :], :], dim = 1, p=2)
    factor = 1 / ((4 * math.pi * self.epsilon ** 2) ** (d / 2))
    exponent = torch.exp(-sq_dist / (4 * self.epsilon ** 2))
    kernel = factor * exponent  # [10138]
    return kernel