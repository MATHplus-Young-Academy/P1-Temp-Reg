from sirf.Gadgetron import ImageData
from sirf.Utilities import assert_validity

import numpy as np

class Gradient1D:
    
    def __init__(self, axis):
        self.axis = axis

    def forward(self, u ):
        assert_validity(u, ImageData)
        u = u.copy().fill(np.roll(u.as_array(), -1, axis=self.axis) - u.as_array())
        
        return u

    def backward(self, u ):
        assert_validity(u, ImageData)
        u = u.copy().fill(np.roll(u.as_array(), 1, axis=self.axis) - u.as_array())
        
        return u    

class Dx():
    def __init__(self):
        self.Gx = Gradient1D(axis=1)
        self.Gy = Gradient1D(axis=2)
        
    def forward(self, u):
        
        dx = self.Gx.forward(u)
        dy = self.Gy.forward(u)
        
        return (dx,dy)
    
    def backward(self, u):
        
        rv = self.Gx.backward(u[0])
        rv += self.Gy.backward(u[1])
    
        return rv
    
class Dt():

    def __init__(self):
        self.Gt = Gradient1D(axis=0)

    def forward(self, u):
        return self.Gt.forward(u)
    def backward(self, u):
        return self.Gt.backward(u)
    
    