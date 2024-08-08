import torch
from torch import nn
import torch_sparse
from torch.nn.init import uniform, xavier_uniform_
from base_classes import ODEFunc
from utils import MaxNFEException
import numpy as np
from torch_geometric.utils import softmax
import math

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

    self.reaction_tanh = False
    self.epsilon = 1.0  # gaussian kernel variance (取为0.1时会梯度爆炸 loss=nan)
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
  
  def calculate_log_kernel(self, x):  # x: [2485, 64]
    Log_Kernel = SpGraphlogKernelLayer(self.in_features, self.out_features, self.opt, self.device).to(self.device)
    n = x.shape[0]  # number of nodes
    k = Log_Kernel(x, self.edge_index)  # [10138]
    kx = torch_sparse.spmm(self.edge_index, k, n, n, x) # [2485, 64]
    return kx 
    
  def calculate_gat_kernel(self, x):
    GAT_Kernel = SpGraphAttentionLayer(self.in_features, self.out_features, self.opt, self.device).to(self.device)
    k = GAT_Kernel(x, self.edge_index)  # torch.Size([10138, 1])
    # 使用更高效的计算方式代替 torch.stack 和 torch_sparse.spmm
    kx = torch.zeros_like(x)
    for idx in range(k.shape[1]):
        kx += torch_sparse.spmm(self.edge_index, k[:, idx], x.shape[0], x.shape[0], x)
    kx /= k.shape[1]  # 取平均值
    return kx
  
  def calculate_gauss_kernel(self, x):
    n = x.shape[0]  # number of nodes 2485
    Gauss_Kernel = SpGraphgaussKernelLayer(self.in_features, self.out_features, self.opt, self.device).to(self.device)
    k = Gauss_Kernel(x, self.edge_index)  # torch.Size([10138])
    kx = torch_sparse.spmm(self.edge_index, k, n, n, x) # [2485, 64]
    return kx 

  def forward(self, t, x):  # the t param is needed by the ODE solver.
    if self.nfe > self.opt["max_nfe"]:  
      raise MaxNFEException
    self.nfe += 1
    if not self.opt['no_alpha_sigmoid']:
      alpha = torch.sigmoid(self.alpha_train)
      beta = torch.sigmoid(self.beta_train)
    else:
      alpha = self.alpha_train
      beta = self.beta_train

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
    elif self.opt['reaction_term'] =='x':
      reaction = x
    elif self.opt['reaction_term'] =='Ax':
      reaction = self.sparse_multiply(x)
    elif self.opt['reaction_term'] =='AAx':
      reaction = self.sparse_multiply(x)
      reaction = self.sparse_multiply(reaction)
      
    elif self.opt['reaction_term'] =='negx':
      reaction = -x
    elif self.opt['reaction_term'] =='negAx':
      reaction = -self.sparse_multiply(x)
    elif self.opt['reaction_term'] =='negAAx':
      reaction = self.sparse_multiply(x)
      reaction = -self.sparse_multiply(reaction)
    elif self.opt['reaction_term'] =='x-Ax':
      reaction = -diffusion
    elif self.opt['reaction_term'] =='Ax-x':
      reaction = diffusion
    elif self.opt['reaction_term'] =='x-Ax+AAx':
      reaction =  -self.sparse_multiply(diffusion) + x
    elif self.opt['reaction_term'] =='Ax-x-AAx':
      reaction = self.sparse_multiply(diffusion) - x
    elif self.opt['reaction_term'] =='zero':
      reaction = 0.0
    
    # aggeragation diffusion term
    elif self.opt['reaction_term'] =='aggdiff-log':
      kx = self.calculate_log_kernel(x)  # torch.Size([2485, 2485])
      reaction = 1e-4*(ax-x)*kx  # 1e-4 is a hyperparameter to avoid gradient explosion
      
    elif self.opt['reaction_term'] =='aggdiff-gat':
      kx = self.calculate_gat_kernel(x)  # torch.Size([2485, 64])
      # print(f'After calculate_gat_kernel: {torch.cuda.memory_allocated() / 1024 ** 2} MB')  # monitor GPU memory usage
      reaction =  1e-4*(ax-x)*kx
      
    elif self.opt['reaction_term'] =='aggdiff-gauss':
      kx = self.calculate_gauss_kernel(x)  # torch.Size([2485, 2485])
      reaction = 1e-4*(ax-x)*kx  # 1e-4 is a hyperparameter to avoid gradient explosion
      
    
       
    elif self.opt['reaction_term'].split('_')[0] == 'exp':
      orders = int(self.opt['reaction_term'].split('_')[1])
      reaction = x
      if orders > 0:
        high_orders = [reaction]
        for order in range(1, orders+1):
          high_order = self.sparse_multiply(high_orders[-1])
          reaction = reaction + 1/np.prod(range(1,order+1)) * high_order
          high_orders.append(high_order)
      
          
    elif self.opt['reaction_term'].split('_')[0] == 'expn':
      orders = int(self.opt['reaction_term'].split('_')[1])
      reaction = x
      if orders > 0:
        high_orders = [reaction]
        for order in range(1, orders+1):
          high_order = self.sparse_multiply(high_orders[-1])
          reaction = reaction + (-1)**order/np.prod(range(1,order+1)) * high_order
          high_orders.append(high_order)

    elif self.opt['reaction_term'].split('_')[0] == 'log':
      orders = int(self.opt['reaction_term'].split('_')[1])
      reaction = self.sparse_multiply(x)
      if orders > 0:
        high_orders = [reaction]
        for order in range(1, orders+1):
          high_order = self.sparse_multiply(high_orders[-1])
          reaction = reaction + (-1)**order/(order+1) * high_order
          high_orders.append(high_order)
          
          
    elif self.opt['reaction_term'].split('_')[0] == 'sin':
      orders = int(self.opt['reaction_term'].split('_')[1])
      reaction = self.sparse_multiply(x)
      if orders > 0:
        high_orders = [reaction]
        for order in range(1, orders+1):
          for _ in range(2):
            high_order = self.sparse_multiply(high_orders[-1])
            high_orders.append(high_order)
          reaction = reaction + (-1)**order/np.prod(order*2+1) * high_order
          
          
    elif self.opt['reaction_term'].split('_')[0] == 'cos':
      orders = int(self.opt['reaction_term'].split('_')[1])
      reaction = x
      if orders > 0:
        high_orders = [reaction]
        for order in range(1, orders+1):
          for _ in range(2):
            high_order = self.sparse_multiply(high_orders[-1])
            high_orders.append(high_order)
          reaction = reaction + (-1)**order/np.prod(order*2) * high_order
    
    else:
      raise Exception('Unknown reaction term.')

    # define the gread diffusion-reation form using reaction_term and diffusion
    """
    - `f` is the reaction-diffusion form
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
        # print(f'End of forward: {torch.cuda.memory_allocated() / 1024 ** 2} MB')
    elif self.opt['beta_diag'] == True:
      f = alpha*diffusion + reaction@self.Beta  # torch.Size([2485, 64])
      
    else:
      raise Exception('Unknown reaction term.')
    
    """
    - We do not use the `add_source` on GREAD
    """
    if self.opt['add_source']:
      f = f + self.source_train * self.x0
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

    self.W = nn.Parameter(torch.zeros(size=(in_features, self.attention_dim))).to(device)
    nn.init.xavier_normal_(self.W.data, gain=1.414) # torch.Size([64, 64])

    self.Wout = nn.Parameter(torch.zeros(size=(self.attention_dim, self.in_features))).to(device)
    nn.init.xavier_normal_(self.Wout.data, gain=1.414)  # torch.Size([64, 64])

    # self.d_k:16 （每个头的维数）
    self.a = nn.Parameter(torch.zeros(size=(2 * self.d_k, 1, 1))).to(device)
    nn.init.xavier_normal_(self.a.data, gain=1.414) # torch.Size([32, 1, 1])

    self.leakyrelu = nn.LeakyReLU(self.alpha)

  def forward(self, x, edge): # x: torch.Size([2485, 64]), edge: 2 x E  torch.Size([2, 10138])
    wx = torch.mm(x, self.W)  # h: N x out  wx: torch.Size([2485, 64])
    h = wx.view(-1, self.h, self.d_k) # torch.Size([2485, 1, 64])
    h = h.transpose(1, 2) # torch.Size([2485, 64, 1])
    # Self-attention on the nodes - Shared attention mechanism
    # import pdb;pdb.set_trace()
    edge_h = torch.cat((h[edge[0, :], :, :], h[edge[1, :], :, :]), dim=1).transpose(0, 1).to(
     self.device)  # edge: 2*D x E  torch.Size([128, 10138, 1])
    # self.a : torch.Size([128, 1, 1])
    edge_e = self.leakyrelu(torch.sum(self.a * edge_h, dim=0)).to(self.device)  # torch.Size([10138, 1])
    # edge_e = torch.log(torch.norm(h[edge[0, :], :, 0] - h[edge[1, :], :, 0], dim = 1)[...,None])
    attention = softmax(edge_e, edge[self.opt['attention_norm_idx']])
    return attention


class SpGraphlogKernelLayer(nn.Module):
  def __init__(self, in_features, out_features, opt, device):
    super(SpGraphlogKernelLayer, self).__init__()
    self.in_features = in_features  # 64
    self.out_features = out_features  # 64
    self.device = device
    self.opt = opt
    self.epsilon = 1e-4 # avoid nan in log kernel
    # self.W = nn.Parameter(torch.zeros(size=(in_features, self.out_features))).to(device)
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
