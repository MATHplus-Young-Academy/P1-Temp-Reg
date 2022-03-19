from sirf.Gadgetron import AcquisitionData, CoilSensitivityData, AcquisitionModel, ImageData
from sirf.Utilities import assert_validity

import numpy as np
    
def get_test_img():

    filepath_y = "/home/jovyan/InputData/y_4.h5"
    y = AcquisitionData(filepath_y)
    u = ImageData()
    u.from_acquisition_data(y)

    csm = CoilSensitivityData()
    csm.calculate(y)

    A = AcquisitionModel(acqs=y, imgs=u)
    A.set_coil_sensitivity_maps(csm)

    u_test = A.backward(y)

    return u_test

def test_image_operator_adjointness(operator, tol=1e-3):
    
    u = get_test_img()
    u = u.fill(np.random.randn(*u.shape))
    
    Ou = operator.forward(u)
    
    b = Ou.copy()
    b.fill(np.random.randn(*Ou.shape).astype(np.complex64))
    
    OHb = operator.backward(b) 
    
    Ou_b = np.vdot( Ou.as_array(), b.as_array())
    u_OHb = np.vdot( u.as_array(), OHb.as_array())
    
    diff_relative = 2* np.abs(Ou_b - u_OHb) / np.abs(Ou_b + u_OHb)
    print(diff_relative)
    
    return np.abs(diff_relative) < tol

def test_stacked_image_operator_adjointness(operator, tol=1e-2):
    
    u = get_test_img()
    u = u.fill(np.random.randn(*u.shape))
    
    Ou = operator.forward(u)
    
    b0 = Ou[0].copy().fill(np.random.randn(*u.shape))
    b1 = Ou[1].copy().fill(np.random.randn(*u.shape))
    
    b = (b0,b1)
        
    OHb = operator.backward(b) 
    u_OHb = np.vdot( u.as_array(), OHb.as_array())
    
    Ou =  np.stack((Ou[0].as_array()[:], Ou[1].as_array()[:]), axis=0)
    b = np.stack((b[0].as_array()[:], b[1].as_array()[:]), axis=0)
    Ou_b = np.vdot( Ou, b)
    
    diff_relative = 2* np.abs(Ou_b - u_OHb) / np.abs(Ou_b + u_OHb)
    print(diff_relative)
        
    return np.abs(diff_relative) < tol

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

class Gradient1D_Local:
    
    def __init__(self, axis, weights):
        assert_validity(weights, ImageData)
        self.weights=weights
        self.axis = axis

    def forward(self, u):
        assert_validity(u, ImageData)
        return self.weights*u.copy().fill(np.roll(u.as_array(), -1, axis=self.axis) - u.as_array())
    
    def backward(self, u ):
        assert_validity(u, ImageData)
        res = np.roll((self.weights * u).as_array(), 1, axis=self.axis) - (self.weights*u).as_array()
        u = u.copy().fill(res)
        
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

    
# Operator with local weights
class Dx_Local():
    
    def __init__(self, weights):
        self.Gx = Gradient1D_Local(weights=weights,axis=1)
        self.Gy = Gradient1D_Local(weights=weights,axis=2)
    
    def forward(self, u):
        dx = self.Gx.forward(u)
        dy = self.Gy.forward(u)
        
        return (dx,dy)
    
    def backward(self,u):
        rv = self.Gx.backward(u[0])
        rv += self.Gy.backward(u[1])
    
        return rv
    
class Dt_Local():

    def __init__(self, weights):
        self.Gt = Gradient1D_Local(weights=weights, axis=0)
    def forward(self, u):
        return self.Gt.forward(u)
    def backward(self, u):
        return self.Gt.backward(u)
