<div align="center">
    <img src="https://raw.githubusercontent.com/therooler/jaxmg/main/docs/_static/logo.png" alt="Jaxmg" width="300">
</div>

#  JAXMg: A multi-GPU linear solver in JAX

[![Docs](https://img.shields.io/badge/docs-site-blue?style=flat-square)](https://flatironinstitute.github.io/jaxmg/)
[![Releases](https://img.shields.io/github/v/release/therooler/jaxmg?style=flat-square)](https://github.com/therooler/jaxmg/releases)
[![Build Status](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/lastBuild/badge/icon)](https://jenkins.flatironinstitute.org/job/jaxmg/job/main/)


# JAXMg
JAXMg provides a C++ interface between [JAX](https://github.com/google/jax) and [cuSolverMg](https://docs.nvidia.com/cuda/cusolver/index.html#using-the-cuSolverMg-api), NVIDIA’s multi-GPU linear solver.  We provide a jittable API for the following routines.

- [cusolverMgPotrs](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotrs-deprecated): Solves the system of linear equations: $Ax=b$ where $A$ is an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition 
- [cusolverMgPotri](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgpotri-deprecated): Computes the inverse of an $N\times N$ symmetric (Hermitian) positive-definite matrix via a Cholesky decomposition.
- [cusolverMgSyevd](https://docs.nvidia.com/cuda/cusolver/index.html#cusolvermgsyevd-deprecated): Computes eigenvalues and eigenvectors of an $N\times N$ symmetric (Hermitian) matrix.

For more details, see the [API](api/potrs.md).

## Downstream logdet branch

This downstream branch adds native log determinant support to the multi-GPU
`potrs` workflow. In addition to solving `A x = b`, `potrs` can return
`log|A|` from the Cholesky factor already computed by the native solver:

```python
x, logdet = potrs(
    A,
    b,
    T_A=T_A,
    mesh=mesh,
    in_specs=P("x", None),
    return_logdet=True,
)
```

Set `return_status=True` as well to return `(x, logdet, status)`.

## Installation

This repository is a downstream development branch of JAXMg. It is not the
released PyPI package. Installing the package name `jaxmg` from PyPI will install
the upstream release, not the branch-specific log determinant support in this
repository.

There are two supported installation paths for this downstream branch.

### 1. Build from source

Use this path when no prebuilt wheel matches your system, or when you are
developing the native C++/CUDA code. Registering the checkout in the active
Python environment and building the native CUDA libraries are separate steps.
Build the native libraries first, then install the Python package:

```bash
git clone https://github.com/JacobTutt/jaxmg.git
cd jaxmg

mkdir build
cd build
cmake ..
cmake --build . --target install
cd ..

pip install ".[cuda12-local]"
```

The CMake install step builds the native shared libraries into `src/jaxmg/cu12`
or `src/jaxmg/cu13`, depending on the CUDA toolkit used for the build. Any
branch that changes C++ or CUDA sources must rebuild these libraries.

### 2. Prebuilt wheels

Use this path when a wheel is provided for an environment matching your Python,
CUDA, and system architecture. The wheel already contains the native shared
libraries, so users do not need to run CMake locally. If you need to run CMake,
use the source build path above instead.

Included wheels:

- `wheels/csd3/jaxmg-0.0.7-cp311-cp311-linux_x86_64.whl`: built on CSD3
  for Linux x86_64 and CPython 3.11 using CUDA 12.1. It is intended for CSD3
  and systems with a compatible NVIDIA driver and Linux x86_64 software stack.
- `wheels/isambard/jaxmg-0.0.7-cp311-cp311-linux_aarch64.whl`: built on
  Isambard-AI Phase 2 for Linux aarch64 and CPython 3.11 using CUDA 12.6. It
  is intended for Isambard and systems with a compatible NVIDIA driver and
  Linux aarch64 software stack.

To install the included wheel, clone this repository and install from the
relative wheel path:

```bash
git clone https://github.com/JacobTutt/jaxmg.git
cd jaxmg
pip install "wheels/csd3/jaxmg-0.0.7-cp311-cp311-linux_x86_64.whl[csd3]"
```

The `csd3` extra installs the JAX runtime used for this wheel:
`jax[cuda12]==0.10.1`.

For Isambard-compatible ARM64 systems:

```bash
git clone https://github.com/JacobTutt/jaxmg.git
cd jaxmg
pip install "wheels/isambard/jaxmg-0.0.7-cp311-cp311-linux_aarch64.whl[isambard]"
```

The `isambard` extra installs the JAX runtime used for this wheel:
`jax[cuda12]==0.10.1`.

Details for compiling the source code can be found in `CONTRIBUTING.md`.

## Example

A minimal example that runs the code is:

```python
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax.sharding import PartitionSpec as P, NamedSharding
from jaxmg import potrs
print(f"Devices: {jax.devices()}")
# Assumes we have at least one GPU available
devices = jax.devices("gpu")
N = 12
T_A = 3
dtype = jnp.float64
# Create diagonal matrix and `b` all equal to one
A = jnp.diag(jnp.arange(N, dtype=dtype) + 1)
b = jnp.ones((N, 1), dtype=dtype)
ndev = len(devices)
# Make mesh and place data (rows sharded)
mesh = jax.make_mesh((ndev,), ("x",))
A = jax.device_put(A, NamedSharding(mesh, P("x", None)))
b = jax.device_put(b, NamedSharding(mesh, P(None, None)))
# Call potrs
out, logdet = potrs(
    A, b, T_A=T_A, mesh=mesh, in_specs=(P("x", None), ), return_logdet=True
)
print(out)
print(logdet)
expected_out = 1.0 / (jnp.arange(N, dtype=dtype) + 1)
expected_logdet = jnp.sum(jnp.log(jnp.arange(N, dtype=dtype) + 1))
print(jnp.allclose(out.flatten(), expected_out))
print(jnp.allclose(logdet, expected_logdet))

```
which gives
```bash
[[1.        ]
 [0.5       ]
 [0.33333333]
 [0.25      ]
 [0.2       ]
 [0.16666667]
 [0.14285714]
 [0.125     ]
 [0.11111111]
 [0.1       ]
 [0.09090909]
 [0.08333333]]
19.987214495661885
True
True
```
as expected.
## Projects that use JAXMg

- [JAXMg Benchmarks](https://github.com/therooler/jaxmg_benchmark): Benchmarks for various Multi-GPUs setups.
- [JAXMg + Netket](https://github.com/therooler/netket_jaxmg): Implementation of the MinSR Netket driver that uses JAXMg for inverting the S-matrix. Tested on Multi-node settings.
- [JAXMg for blurred sampling](https://github.com/therooler/nqs_blurred_sampling): Implementation of t-VMC that makes use JAXMg for inverting the QGT.

## cuSolverMp
As of CUDA 13, there is a new distributed linear algebra library called [cuSolverMp](https://docs.nvidia.com/cuda/cusolvermp/) with similar capabilities as cuSolverMg, that does support multi-node computations as well as >16 devices. Given the similarities in syntax, it should be straightforward to eventually switch to this API. This will require sharding data into a cyclic 2D form and handling the solver orchestration with MPI.

## Citations
```
@misc{2601.14466,
Author = {Roeland Wiersema},
Title = {JAXMg: A multi-GPU linear solver in JAX},
Year = {2026},
Eprint = {arXiv:2601.14466},
}
```

## Acknowledgements
I acknowledge support from the Flatiron Institute. The Flatiron Institute is a
division of the Simons Foundation.
