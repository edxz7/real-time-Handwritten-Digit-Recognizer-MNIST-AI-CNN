"""
Capsule Networks 
Code adapted from: github.com/gram-ai/capsule-networks
License: Apache-2.0
Author: Eduardo Ch. Colorado
E-mail: edxz7c@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from fastai.vision import *
# from fastai import *

NUM_ROUTING_ITERATIONS = 3
NUM_CLASSES = 10
batch_size=128

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def softmax(input, dim=1):
    transposed_input = input.transpose(dim, len(input.size()) - 1)
    softmaxed_output = F.softmax(transposed_input.contiguous().view(-1, transposed_input.size(-1)), dim=-1)
    return softmaxed_output.view(*transposed_input.size()).transpose(dim, len(input.size()) - 1)



class CapsuleLayer(nn.Module):
  def __init__(self, num_capsules, num_route_nodes, in_channels=256, 
               out_channels=32, kernel_size=9, stride=2, num_iterations=NUM_ROUTING_ITERATIONS):
      super().__init__()

      # num_route_nodes is the vetor input
      self.num_route_nodes = num_route_nodes
      self.num_capsules  = num_capsules
      self.num_iterations= num_iterations

      if self.num_route_nodes != -1: 
        self.W = nn.Parameter(torch.randn(num_capsules, num_route_nodes, in_channels, out_channels)) 
      else:
        self.Capsules = nn.ModuleList([nn.Conv2d(in_channels, out_channels, 
                                             kernel_size, stride, padding=0) 
                                   for _ in range(num_capsules)])
    
    # squashing function
  def squash(self, tensor, dim=-1):
      squared_norm = (tensor ** 2).sum(dim=dim, keepdim=True)
      scale = squared_norm / (1 + squared_norm)
      return scale * tensor / torch.sqrt(squared_norm)
    
  def forward(self, x):
      #################
      # Code for calculate the digit capss output vector
      ################
      if self.num_route_nodes != -1:
        # Each None adds a dimension to the tensor, the : slice all the avaible elements in the current dimensions 
        # u is 3d and here we are adding a dimension for the batch size and another to
        # for the stacked u vectors from the primary caps 
        u = x[None, :, :, None, :]
        # The weights matrix is a rank 4 tensor and we need add an extra dim to make it compatible
        # with the shape of u
        W = self.W[:, None, :, :, :]
        # u_hat are the priors
        u_hat = u @ W
        
        # log prior probabilities (logits),
        b_ij = torch.zeros(*u_hat.size(), device=device)
        #b_ij = Variable(torch.zeros(*u_hat.size())).cuda()

        # Dynamic routing agreement calculation
        for i in range(self.num_iterations):
          # probabilities coupling coefficients
          c_ij = softmax(b_ij, dim=2)
          
          # total capsule input sum(c_ij*u_hat)
          s_ij = (c_ij* u_hat).sum(dim=2, keepdim=True)
          
          # squashing to make compatible the output vector with the coupling loss
          v_j = self.squash(s_ij)
          
          if i < self.num_iterations - 1:
            # agreement
            a_ij = (v_j * u_hat).sum(dim=-1, keepdim=True)
            # update the logits
            b_ij += a_ij
          return v_j
            
      #####
      # code for calculate the output vector u for the primary caps
      #####
      else:
        # get the batch size from input
        batch_size = x.size(0)
        # reshape the convolutional layer outputs to be (batch_size, vector_dim=32*6*6)
        u = [capsule(x).view(batch_size, -1, 1) for capsule in self.Capsules]
        # stack up the output vectors u for each of the capsules
        u = torch.cat(u, dim=-1)
        # squash the stacked vectors
        u = self.squash(u)
        return u  
      
class CapsuleResnetishNet(nn.Module):
  def __init__(self):
    super().__init__()
    
    # 1@28x28
    self.conv1 = nn.Conv2d(in_channels=1, out_channels=256, kernel_size=9, stride=1)
    self.resnet_block = res_block(256)
    self.primary_capsules = CapsuleLayer(num_capsules=8, num_route_nodes=-1, in_channels=256, out_channels=32,
                                         kernel_size=9, stride=2)
    self.digit_capsules = CapsuleLayer(num_capsules=NUM_CLASSES, num_route_nodes=32 * 6 * 6, in_channels=8,
                                       out_channels=16)
    self.decoder = nn.Sequential(
        nn.Linear(16 * NUM_CLASSES, 512),
        nn.ReLU(inplace=True),
        nn.Linear(512, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 784),
        nn.Sigmoid()
    )
    
  def forward(self, x, y=None):
    images = x.clone()
    x = F.relu(self.conv1(x), inplace=True)
    x = self.resnet_block(x)
    x = self.primary_capsules(x)
    if x.size(0) == 1:
      x = self.digit_capsules(x).squeeze().unsqueeze(dim=0)
    else:
      x = self.digit_capsules(x).squeeze().transpose(0, 1)
    
    classes = (x ** 2).sum(dim=-1) ** 0.5
    classes = F.softmax(classes, dim=-1)
    if y is None:
        # In all batches, get the most active capsule.
        _, max_length_indices = classes.max(dim=1)
        
        y = torch.eye(NUM_CLASSES, device=device).index_select(dim=0, index=max_length_indices.data)

    reconstructions = self.decoder((x * y[:, :, None]).view(x.size(0), -1))

    return x, reconstructions, y, images 
     


     
     

def return_model(): 
  return CapsuleResnetishNet()