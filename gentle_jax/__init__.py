"""JAX port of the GENTLE training pipeline.

Import this package before jax so the settings below take effect.

``JAX_DEFAULT_MATMUL_PRECISION=highest`` disables TF32. Without it, matmuls on
Ampere and newer drift ~1e-5 from the torch reference, which is enough to hide
real porting bugs in the parity tests. The networks here are far too small for
the TF32 speedup to matter.

The torch import is a workaround, not a dependency of the port. On this machine
(RTX 5090, sm_120, driver 580, CUDA 12.8 wheels alongside a system CUDA 12.9),
jax initializes its CUDA backend in a state where every matmul fails with
"an unsupported value or parameter was passed to the function" -- at any shape,
including a 32-row Dense. Importing torch beforehand makes jax work; importing
it afterwards does not. Preloading libnvJitLink/libnvrtc, aligning the ptxas
version, unsetting CUDA_HOME, and disabling XLA autotuning and command buffers
were all tried and none of them help, so the cause is something torch does
while initializing its CUDA runtime rather than a missing library.

It is deliberately best-effort: where jax works on its own -- notably the MJX
machine this port targets -- torch need not be installed, and this becomes a
no-op. Delete it once jax no longer needs the help.
"""
import os

os.environ.setdefault('JAX_DEFAULT_MATMUL_PRECISION', 'highest')

try:
    import torch  # noqa: F401
except ImportError:
    pass
