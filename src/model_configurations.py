from function_laplacian_diffusion import LaplacianODEFunc
from block_constant import ConstantODEblock
from block_attention import AttODEblock
from function_gread import ODEFuncGread
from function_diffusion_aggregation import DiffusionAggregationODEFunc

class BlockNotDefined(Exception):
  pass

class FunctionNotDefined(Exception):
  pass

def set_block(opt):
  ode_str = opt['block']
  if ode_str == 'constant':
    block = ConstantODEblock
  elif ode_str == 'attention':
    block = AttODEblock
  else:
    raise BlockNotDefined
  return block


def set_function(opt):
  ode_str = opt['function']
  if ode_str == 'laplacian':
    f = LaplacianODEFunc
  elif ode_str == 'gread':
    f = ODEFuncGread
  elif ode_str == 'diffagg':
    f = DiffusionAggregationODEFunc
  else:
    raise FunctionNotDefined
  return f
