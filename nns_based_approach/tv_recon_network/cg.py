import torch
from scipy.sparse.linalg import LinearOperator, cg
from typing import Callable
from torch import Tensor

class CG(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z: Tensor, AcquisitionModel, beta: Tensor, y, G:Callable, GHG:Callable) -> Tensor:
        tmp = AcquisitionModel.adjoint(y)
        b = tmp.as_array().ravel() + (beta * G(z)).numpy().ravel()

        def AHA(x):
            tmp.fill(x)
            return (
                AcquisitionModel.adjoint(AcquisitionModel.direct(tmp))
                .as_array()
                .ravel()
            )

        H = LinearOperator(
            shape=(np.prod(b.shape), np.prod(b.shape)),
            matvec=lambda x: AHA(x) + (beta * GHG(x)).numpy().ravel(),
        )
        sol = cg(H, b)
        xprime = sol[0].reshape(tmp.shape)
        ctx.H = H
        ctx.G = G
        xprime_tensor = torch.from_numpy(xprime)
        ctx.save_for_backward(beta, xprime_tensor)
        return xprime_tensor

    @staticmethod
    def backward(ctx, grad_output):
        beta, xprime = ctx.saved_tensors
        grad = torch.from_numpy(cg(ctx.H, grad_output.numpy().ravel())[0]).reshape(
            grad_output.shape
        )
        gz = beta * ctx.G(grad)
        gbeta = None  # TODO
        return gz, None, gbeta, None, None, None
