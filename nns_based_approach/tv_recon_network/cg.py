import torch
from scipy.sparse.linalg import LinearOperator, cg
from typing import Callable, Optional
from torch import Tensor
import numpy as np

class CG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, AcquisitionModel, beta: Tensor, y, G: Callable, GH: Callable, GHG: Optional[Callable]=None) -> Tensor:
        tmp = AcquisitionModel.adjoint(y)
        if GHG is None: 
            GHG = lambda x: GH(G(x))
        b = tmp.as_array().ravel() + (beta * GH(z)).numpy().ravel()
        def AHA(x):
            tmp.fill(x)
            return AcquisitionModel.adjoint(AcquisitionModel.direct(tmp)).as_array().ravel()
        H = LinearOperator(
            shape=(np.prod(b.shape), np.prod(b.shape)),
            dtype=np.complex64,
            matvec=lambda x: AHA(x)+(beta * GHG(torch.from_numpy(x).reshape(tmp.shape).unsqueeze(0))).numpy().ravel()
        )
        sol = cg(H, b)
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
        grad = torch.from_numpy(cg(ctx.H, grad_output.numpy().ravel())[0]).reshape(grad_output.shape)
        gz = gbeta = None
        if ctx.needs_input_grad[0]:
            gz = beta * ctx.G(grad)
        if ctx.needs_input_grad[2]:
            gbeta = (-ctx.GH(ctx.G(xprime) - z) * grad).sum().real
        return gz, None, gbeta, None, None, None
