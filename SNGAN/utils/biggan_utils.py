

import random

from utils.sample import sample_latents

import torch



class ema_(object):
  def __init__(self, source, target, decay=0.9999, start_itr=0):
    self.source = source
    self.target = target
    self.decay = decay
    # Optional parameter indicating what iteration to start the decay at
    self.start_itr = start_itr
    # Initialize target's params to be source's
    self.source_dict = self.source.state_dict()
    self.target_dict = self.target.state_dict()
    print('Initializing EMA parameters to be source parameters...')
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.source_dict[key].data)
        # target_dict[key].data = source_dict[key].data # Doesn't work!

  def update(self, itr=None):
    # If an iteration counter is provided and itr is less than the start itr,
    # peg the ema weights to the underlying weights.
    if itr and itr < self.start_itr:
      decay = 0.0
    else:
      decay = self.decay
    with torch.no_grad():
      for key in self.source_dict:
        self.target_dict[key].data.copy_(self.target_dict[key].data * decay + self.source_dict[key].data * (1 - decay))


def ortho(model, strength=1e-4, blacklist=[]):
  with torch.no_grad():
    for param in model.parameters():
      # Only apply this to parameters with at least 2 axes, and not in the blacklist
      if len(param.shape) < 2 or any([param is item for item in blacklist]):
        continue
      w = param.view(param.shape[0], -1)
      grad = (2 * torch.mm(torch.mm(w, w.t())
              * (1. - torch.eye(w.shape[0], device=w.device)), w))
      param.grad.data += strength * grad.view(param.shape)


# Interp function; expects x0 and x1 to be of shape (shape0, 1, rest_of_shape..)
def interp(x0, x1, num_midpoints):
    lerp = torch.linspace(0, 1.0, num_midpoints + 2, device='cuda').to(x0.dtype)
    return ((x0 * (1 - lerp.view(1, -1, 1))) + (x1 * lerp.view(1, -1, 1)))
