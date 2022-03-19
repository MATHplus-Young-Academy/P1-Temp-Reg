import numpy as np

class A :
    
    def __init__(self, mask):
        self.mask = mask
        
    def forward(self,y):
        return self.mask * np.fft.fftshift( \
        np.fft.ifftn( \
        np.fft.ifftshift(y, axes=(0,1)),axes=(0,1),norm='ortho'),axes=(0,1))
    
    def backward(self,x):
        return np.fft.ifftshift( \
        np.fft.fftn( \
        np.fft.fftshift(self.mask*x, axes=(0,1)),axes=(0,1),norm='ortho'),axes=(0,1))
        
class Grad1D:
    
    def __init__(self,axis):
        self.axis=axis
    
    def forward(self,x):
        return np.roll(x, -1, axis=self.axis) - x
    def backward(self,x):
        return np.roll(x, 1, axis=self.axis) - x

class Dx:
    def __init__(self):
        self.Gx = Grad1D(axis=0)
        self.Gy = Grad1D(axis=1)
        
    def forward(self, x):
        return (self.Gx.forward(x), self.Gy.forward(x))
        
    def backward(self, x):
        xt = self.Gx.backward(x[0])
        xt += self.Gy.backward(x[1])
        return xt

class Dt:
    def __init__(self):
        self.Gt = Grad1D(axis=2)
        
    def forward(self, x):
        return self.Gt.forward(x)
        
    def backward(self, x):
        return self.Gt.backward(x)
