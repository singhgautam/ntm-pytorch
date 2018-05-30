import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

class ExternalMemory(nn.Module):
    def __init__(self,
                 mem_size,
                 mem_width,
                 device):

        #parameter initialization
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.device = device

        #make memory
        self.mem = torch.randn(mem_size, mem_width, dtype=torch.double).to(self.device)



