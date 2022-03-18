import torch
from scipy.sparse.linalg import LinearOperator, cg
from typing import Callable, Optional
from torch import Tensor
import numpy as np
import time

class CG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, AcquisitionModel, beta: Tensor, y, G: Callable, GH: Callable, GHG: Optional[Callable]=None,x0:Optional[Tensor]=None) -> Tensor:
        tmp = AcquisitionModel.adjoint(y)
        if GHG is None: 
            GHG = lambda x: GH(G(x))
        b = tmp.as_array().ravel() + (beta * GH(z)).numpy().ravel()
        if x0 is not None:
            x0 = x0.numpy().ravel()
        def AHA(x):
            tmp.fill(x)
            return AcquisitionModel.adjoint(AcquisitionModel.direct(tmp)).as_array().ravel()
        H = LinearOperator(
            shape=(np.prod(b.shape), np.prod(b.shape)),
            dtype=np.complex64,
            matvec=lambda x: AHA(x)+(beta * GHG(torch.from_numpy(x).reshape(tmp.shape).unsqueeze(0))).numpy().ravel()
        )
        sol = cg(H, b,tol=1e-3,x0=x0)
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
        print('backward cg',time.time()-old)
        gz = gbeta = None
        if ctx.needs_input_grad[0]:
            gz = beta * ctx.G(grad.unsqueeze(0))
        if ctx.needs_input_grad[2]:
            gbeta = (-ctx.GH(ctx.G(xprime.unsqueeze(0)) - z.unsqueeze(0)) * grad).sum().real
        return gz, None, gbeta, None, None, None, None, None
