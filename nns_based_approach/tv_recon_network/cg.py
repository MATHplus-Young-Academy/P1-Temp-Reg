import torch
from scipy.sparse.linalg import LinearOperator, cg
from typing import Callable, Optional
from torch import Tensor
import numpy as np
import time


class Hop:
    def __init__(self,template,AcquisitionModel,GHG,beta):
        self.template = template.clone()
        self.AcquisitionModel=AcquisitionModel
        self.GHG=GHG
        self.beta = beta
    def __call__(self,x):
        self.template.fill(x)
        AHA=self.AcquisitionModel.adjoint(self.AcquisitionModel.direct(self.template)).as_array().ravel()
        GHG=(self.beta * self.GHG(torch.from_numpy(x).reshape(self.template.shape).unsqueeze(0))).numpy().ravel()
        return AHA+GHG
    
class CG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, AcquisitionModel, beta: Tensor, y, G: Callable, GH: Callable, GHG: Optional[Callable]=None,H_op=None,x0:Optional[Tensor]=None) -> Tensor:
        tmp = AcquisitionModel.adjoint(y)
        if GHG is None: 
            GHG = lambda x: GH(G(x))
        if H_op is None:
             H_op=Hop(tmp,acq_model,self.GOps.apply_GHG , F.softplus(self.beta_reg,beta=10.))
        b = tmp.as_array().ravel() + (beta * GH(z)).numpy().ravel()
        if x0 is not None:
            x0 = x0.numpy().ravel()
        
        H = LinearOperator(
            shape=(np.prod(b.shape), np.prod(b.shape)),
            dtype=np.complex64,
            matvec=H_op
            #lambda x: AHA(x)+(beta * GHG(torch.from_numpy(x).reshape(tmp.shape).unsqueeze(0))).numpy().ravel()
        )
        sol = cg(H, b,tol=1e-5,x0=x0)
        xprime = sol[0].reshape(tmp.shape)
        ctx.H = H
        ctx.G = G
        ctx.GH = GH
        xprime_tensor = torch.from_numpy(xprime)
        ctx.save_for_backward(beta, xprime_tensor, z)
        return xprime_tensor

    @staticmethod
    def backward(ctx, grad_output):
        beta, xprime, z = ctx.saved_tensors
        b = grad_output.unsqueeze(0).numpy().ravel()
        old=time.time()
        grad = torch.from_numpy(cg(ctx.H, b,tol=1e-3, x0=b)[0]).reshape(grad_output.shape)
        print('backward cg time',time.time()-old)
        gz = gbeta = None
        if ctx.needs_input_grad[0]:
            gz = beta * ctx.G(grad.unsqueeze(0))
        if ctx.needs_input_grad[2]:
            gbeta = (-ctx.GH(ctx.G(xprime.unsqueeze(0)) - z) * grad).sum().real
        return gz, None, gbeta, None, None, None, None, None, None
