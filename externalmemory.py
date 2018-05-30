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

        #shift matrices
        shiftNone = torch.diag(torch.ones(mem_size)).to(self.device)
        shiftLeft = torch.zeros(mem_size,mem_size).to(self.device)
        shiftRight = torch.zeros(mem_size,mem_size).to(self.device)
        for i in range(mem_size):
            shiftLeft[i,(i+1)%mem_size] = 1.0
        for i in range(mem_size):
            shiftLeft[i,(i-1)%mem_size] = 1.0



    ''' 
    Reads values from memory
    using w as the read head.
    Read head is a vector of
    weights on each memory location.
    '''
    def read(self, w):
        return torch.matmul(torch.transpose(self.mem,0,1),w)

    '''
    Erases the tensor from the memory
    addressed by w and eraser given by e.  
    '''
    def erase(self, w, e):
        self.mem = self.mem*[1 - w*e]

    '''
    Inserts tensor 'a' at the memory location
    addressed by w
    '''
    def add(self, w, a):
        self.mem = self.mem + w*a

    '''
    Find a w-address of the relevant memory using
    the input key and beta key-strength. For beta -> 0
    the softmax nature of addressing is softened.
    '''
    def address_by_content(self, key, beta):
        K = torch.matmul(torch.transpose(self.mem,0,1),key)
        expK = torch.exp(beta*K)
        return expK/(torch.sum(expK))

    '''
    
    '''
    def address_by_location(self, ):