import math

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F




# ------full factorization entropy model

class Bitparm(nn.Module):
    '''
    save params
    '''
    def __init__(self, channel, final=False):
        super().__init__()
        self.final = final
        self.h = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        self.b = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        if not final:
            self.a = nn.Parameter(torch.nn.init.normal_(torch.empty(channel).view(1, -1, 1, 1), 0, 0.01))
        else:
            self.a = None

    def forward(self, x):
        if self.final:
            # return F.sigmoid(x * F.softplus(self.h) + self.b)
            return torch.sigmoid(x * F.softplus(self.h) + self.b)
        else:
            x = x * F.softplus(self.h) + self.b
            # return x + F.tanh(x) * F.tanh(self.a)
            return x + torch.tanh(x) * torch.tanh(self.a)
        

class BitEstimator(nn.Module):  # return the probability of channel
    '''
    Estimate bit
    '''
    def __init__(self, channel):
        super().__init__()
        self.f1 = Bitparm(channel)
        self.f2 = Bitparm(channel)
        self.f3 = Bitparm(channel)
        self.f4 = Bitparm(channel, True)
        
    def forward(self, x):
        x = self.f1(x)
        x = self.f2(x)
        x = self.f3(x)
        return self.f4(x)
    
# ------full factorization entropy model end------



if __name__ == "__main__":

   
    print("end")