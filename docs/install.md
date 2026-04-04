# Installation
The package is available on PyPi and can be installed with

```bash
pip install jaxmg[cuda12]
```

This will install a GPU compatible version of JAX. 

1. `pip install "jaxmg[cuda12]"`: Use CUDA 12 (only works for `jax>=0.6.2`).

2. `pip install "jaxmg[cuda12-local]"`: Use locally available CUDA 12 installation.

3. `pip install "jaxmg[cuda13]"`: Use CUDA 13 (only works for `jax>=0.7.2`).

4. `pip install "jaxmg[cuda13-local]"`: Use locally available CUDA 13 installation.

The provided binaries are compiled with

|**JAXMg** | **CUDA** | **cuDNN** |
|---|---|---| 
| `cuda12`,`cuda12-local` | 12.8.0 | 9.17.1.4|
| `cuda13`,`cuda13-local` | 13.0.0 | 9.17.1.4|

> **_Note:_** `pip install jaxmg` will install a CPU-only version of JAX. Since `jaxmg` is a GPU-only package you will receive a warning to install a GPU-compatible version of jax. 

## Source builds on Isambard-like systems

On systems where the packaged wheel layout does not match the local CUDA stack,
`jaxmg` may need to be built from source and pointed at the system `cusolverMg`
library explicitly. The runtime loader in `src/jaxmg/_setup.py` now checks
`JAXMG_CUSOLVERMG_LIB` and `CUSOLVERMG_LIB` before falling back to packaged
NVIDIA Python libraries, which is useful for Isambard-style deployments.

Typical runtime overrides look like:

```bash
export CUSOLVERMG_LIB=/path/to/libcusolverMg.so
export JAXMG_CUSOLVERMG_LIB=$CUSOLVERMG_LIB
export PYTHONPATH=/path/to/jaxmg/src:${PYTHONPATH:-}
```

The current runtime also tolerates a missing `libcusolverMg.so.12` when probing
optional CUDA 13 packaged libraries from a CUDA 12 environment.

## Future cuSOLVERMp note

These installation notes describe the current `cuSolverMg` backend. A future
move to `cuSOLVERMp` would change the installation and runtime model
substantially: expect `cuSOLVERMp`, `NCCL`, and likely `MPI`, plus one
process per GPU and a 2D block-cyclic matrix layout.
