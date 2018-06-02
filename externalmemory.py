import torch
import torch.nn as nn
import torch.utils
import torch.utils.data

class ExternalMemory(nn.Module):
    def __init__(self,
                 mem_size,
                 mem_width,
                 device):
        super(ExternalMemory, self).__init__()
        #parameter initialization
        self.mem_size = mem_size
        self.mem_width = mem_width
        self.device = device

        #make memory
        self.mem = torch.randn(mem_size, mem_width).type(torch.FloatTensor).to(self.device)

        #shift matrices
        self.shiftNone = torch.diag(torch.ones(mem_size)).to(self.device)
        self.shiftLeft = torch.zeros(mem_size,mem_size).to(self.device)
        self.shiftRight = torch.zeros(mem_size,mem_size).to(self.device)
        for i in range(mem_size):
            self.shiftLeft[i,(i+1)%mem_size] = 1.0
        for i in range(mem_size):
            self.shiftRight[i,(i-1)%mem_size] = 1.0

        #softmax
        self.softmax = nn.Softmax(dim=0).to(self.device)



    ''' 
    Reads values from memory
    using w as the read head.
    Read head is a vector of
    weights on each memory location.
    '''
    def read(self, w):
        w = torch.clamp(w,min=0.0, max=1.0)
        return torch.matmul(torch.transpose(self.mem,0,1),w)

    '''
    Erases the tensor from the memory
    addressed by w and eraser given by e.  
    '''
    def erase(self, w, e):
        w = torch.clamp(w, min=0.0, max=1.0)
        e = torch.clamp(e,min=0.0, max=1.0)
        self.mem = self.mem*(1 - torch.ger(w,e))

    '''
    Inserts tensor 'a' at the memory location
    addressed by w
    '''
    def add(self, w, a):
        w = torch.clamp(w, min=0.0, max=1.0)
        a = torch.clamp(a, min=0.0, max=1.0)
        self.mem = self.mem + torch.ger(w,a)

    '''
    Find a w-address of the relevant memory using
    the input key and beta key-strength. For beta -> 0
    the softmax nature of addressing is softened.
    '''
    def address_by_content(self, key, beta):
        key = torch.clamp(key, min=0.0, max=1.0)
        mem_norm = torch.sqrt(torch.sum(torch.pow(self.mem,2),1))
        # print 'outer', torch.ger(mem_norm, torch.ones(self.mem_width)).size()
        K = torch.matmul(self.mem,key)/(mem_norm*torch.norm(key,p=2))
        return self.softmax(beta*K)

    '''
    Find address when the w-address needs to be shifted according 
    to the shift information provided by shift
    '''
    def address_by_location(self, w, shift):
        w = torch.clamp(w, min=0.0, max=1.0)
        shift = torch.clamp(shift, min=0.0, max=1.0)
        shift_conv = shift[0]*self.shiftLeft+shift[1]*self.shiftNone+shift[2]*self.shiftRight
        return torch.matmul(shift_conv, w)

    '''
    Combine addresses
    '''
    def address(self, key, beta, gamma, gate, shift, _w):
        key = torch.clamp(key, min=0.0, max=1.0)
        # gate = torch.clamp(gate, min=0.0, max=1.0)
        shift = torch.clamp(shift, min=0.0, max=1.0)
        _w = torch.clamp(_w, min=0.0, max=1.0)
        # gamma = torch.clamp(gamma, min=1.0)

        wc = self.address_by_content(key, beta)
        wg = gate*wc + (1-gate)*_w
        wl = self.address_by_location(wg, shift)
        wlpow = torch.pow(wl, gamma)
        return wlpow/(torch.sum(wlpow))