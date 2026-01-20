---
title: 'JAXMg: A multi-GPU linear solver in JAX'
tags:
  - Python
  - JAX
  - CUDA
  - distributed linear algebra
authors:
  - name: Roeland Wiersema
    orcid: 0000-0002-0839-4265
    corresponding: true # (This is how to denote the corresponding author)
    affiliation: 1 # (Multiple affiliations must be quoted)
affiliations:
 - name: Center for Computational Quantum Physics, Flatiron Institute, 162 Fifth Avenue, New York, NY 10010, USA
   index: 1
   ror: 00sekdz59
date: 13 January 2026
bibliography: paper.bib

---

# Summary

Solving large dense linear systems and eigenvalue problems is a core requirement in many areas of scientific computing, but scaling these operations beyond a single GPU remains challenging within modern accelerator-centric programming frameworks. While highly optimized multi-GPU solver libraries exist, they are typically difficult to integrate into composable, just-in-time (JIT) compiled Python workflows.

JAXMg provides multi-GPU dense linear algebra for JAX, enabling Cholesky-based linear solves and symmetric eigendecompositions for matrices that exceed single-GPU memory limits. By interfacing JAX with NVIDIA’s cuSOLVERMg through an XLA Foreign Function Interface, JAXMg exposes distributed GPU solvers as JIT-compatible JAX primitives. This design allows scalable linear algebra to be embedded directly within JAX programs, preserving composability with JAX transformations and enabling multi-GPU execution in end-to-end scientific workflows.

# Statement of need

Modern scientific computing increasingly relies on GPUs, which now provide a large fraction of the
available floating-point throughput in both leadership-class supercomputers and smaller multi-GPU
workstations. At the same time, dense linear algebra remains a critical building block for many
numerical methods. A number of mature libraries therefore provide high-performance linear algebra on
CPUs and GPUs, including distributed and accelerator-aware packages such as ScaLAPACK
[@blackford1997scalapack], MAGMA [@abdelfattah2024magma], and SLATE [@gates2019slate].

In parallel, JAX [@jax2018github] has become a widely adopted framework for scientific computing
because it combines a NumPy-like user experience with JIT compilation and automatic
differentiation. The JAX ecosystem has expanded rapidly, with libraries for neural networks
[@flax2020github], Bayesian inference [@cabezas2024blackjax], differential equations [@kidger2021on],
Variational Monte Carlo [@netket3:2022]
and full physics simulation environments [@brax2021github]. These workflows often require repeatedly
solving linear systems or computing eigenvalue decompositions, either as part of a larger simulation
loop or inside differentiable optimization.

Despite this growth, the JAX ecosystem lacks distributed dense linear algebra routines that scale
across multiple GPUs while remaining usable from idiomatic JAX programs. This gap makes it
challenging to take existing JAX-based scientific applications to problem sizes that exceed a single
GPU, or to integrate multi-GPU linear algebra into end-to-end JAX pipelines. 
Existing approaches require leaving the JAX execution model, either by exporting arrays to external MPI-based solvers or by manually orchestrating GPU kernels outside JAX’s JIT. These approaches break composability
and complicate memory management.

JAXMg addresses this need by providing a distributed multi-GPU interface to GPU-accelerated solver
backends, enabling scalable linear solves and eigendecompositions from within JAX. 

# Software design

JAXMg connects JAX to NVIDIA’s multi-GPU dense linear algebra library cuSOLVERMg [@cusolver] via an
XLA Foreign Function Interface (FFI) C++ extension. This design enables writing complex,
JIT-compatible JAX programs while delegating the compute-intensive and communication-sensitive
components to a compiled backend.

The current release provides a jittable interface to the main cuSOLVERMg routines `potrs`, `potri`, and
`syevd`:

- `cusolverMgPotrs`: Solves $Ax=b$ for symmetric (Hermitian) positive-definite $A$ using a Cholesky factorization. 
- `cusolverMgPotri`: Computes the inverse of a symmetric (Hermitian) positive-definite matrix from its Cholesky factorization.
- `cusolverMgSyevd`: Computes eigenvalues and eigenvectors of a symmetric (Hermitian) matrix.

All routines support JAX dtypes `float32`, `float64`, `complex64`, and `complex128`. JAXMg supports
CUDA 12 and CUDA 13 compatible devices.

For example, the `potrs` routine can be called like this over a 1D mesh `mesh = jax.make_mesh((jax.device_count(),), ("x",))`:

```python
out = potrs(A, b, T_A=T_A, mesh=mesh, in_specs=(P("x", None), P(None, None)))
```
Here, `A` is an $N\times N$ positive-definite matrix that is sharded row-wise over the available
devices: `jax.device_put(A, NamedSharding(mesh, P("x", None)))`. The right-hand side `b` has shape
$N \times N_{\mathrm{RHS}}$ and is replicated across devices:
`jax.device_put(b, NamedSharding(mesh, P(None, None)))`. The tile size $T_A$ is user-configurable and
controls the trade-off between memory usage and performance; larger tiles typically improve
throughput (see Benchmark).

## 1D Cyclic Data Distribution

Parallelized linear algebra algorithms require a distributed data layout to ensure proper load
balancing of the available computational power [@dongarra1994]. JAXMg constructs the required 1D
block-cyclic layout in the C++ backend.

In this 1D scheme, columns are assigned to GPUs in fixed-size tiles of $T_A$ columns, distributed in round-robin order across the available devices. Given a global matrix dimension $N$, we first construct an explicit mapping from each global source column index to its destination column index in the target 1D block-cyclic layout.

![Arrays are row-wise sharded and put into the round-robin 1D cyclic form illustrated here.\label{fig:jaxmg_cyclic}](figures/jaxmg_cyclic.png){ width=100% }

To apply this redistribution efficiently in-place, we decompose the column-index mapping into disjoint permutation cycles. Each cycle specifies a sequence of columns that must be rotated to reach the target layout, and we execute these rotations using peer-to-peer GPU copies (e.g., `cudaMemcpyPeerAsync`) together with two small staging buffers to avoid overwriting data before it is forwarded. This yields a deterministic procedure for converting between contiguous per-device column storage and the 1D block-cyclic distribution required by cuSOLVERMg. See \autoref{fig:jaxmg_cyclic} for a schematic depiction.

## Memory management

For parallel execution, JAXMg supports Single Program Multiple Devices (SPMD) and Multi Program
Multiple Devices (MPMD) modes. In both cases, we use the `jax.shard_map` primitive to expose each
device’s local shard and pass the corresponding GPU pointers into the backend.

At execution time, `shard_map` launches one thread (SPMD) or one process (MPMD) per GPU. However, the
cuSOLVERMg API must be called from a single thread/process that can access all GPU pointers. 
The main technical challenge is reconciling JAX’s execution model with cuSOLVERMg’s single-caller requirement.”

In the SPMD case, all threads share a single virtual address space, so sharing pointers is
straightforward: we create a POSIX shared-memory region that stores the per-device pointers
and each thread assigns its respective shard to the shared array.

In the MPMD case, each process has its own virtual address space, and directly sharing device
pointers across processes is undefined. CUDA therefore provides low-level inter-process
communication via the `cudaIpc` API, which allows GPU allocations to be shared between processes. We
illustrate the SPMD and MPMD approaches schematically in \autoref{fig:jaxmg_shm}.

![(Left) In SPMD mode, each GPU is controlled by a separate thread. Since these threads share the same virtual memory space, we can share device pointers between them. (Right) In MPMD mode, we use the `cudaIpc` API to transport the device pointers to a single array in process 0.\label{fig:jaxmg_shm}](figures/jaxmg_shm.png){ width=100% }

# Benchmark

To assess performance, we benchmark JAXMg against the single-GPU routines currently available in JAX.
All experiments are run on a single node with 8 NVIDIA H200 GPUs (143 GB VRAM each) connected via NVLink.

We report three representative cases: `potrs` (float32), `syevd` (float64), and `potri` (complex128).
In all cases we use a diagonal matrix $A=\mathrm{diag}(1,\ldots,N)$[^1], and for `potrs` we set $b=(1,\ldots,1)^\mathsf{T}$.
We vary the tile size $T_A$ and report wall-clock timings in \autoref{fig:benchmark}; the benchmark code is available at [@jaxmg_benchmark].
We see that JAXMg scales better than the native single-GPU linear algebra routines and surpasses them in performance, especially for larger matrices.
Both `syevd` and `potri` require significantly more workspace memory than `potrs`, which is reflected in the matrix sizes that can be reached.

![Benchmark comparing the native single-GPU JAX routines (which call cuSOLVERDn) to JAXMg. Reported timings include array allocation, which is negligible compared to the runtime. Larger tile sizes improve performance only once the problem size is sufficiently large, consistent with a GPU-utilization effect. Tile size has negligible impact for `syevd`, while `potri` shows a strong dependence on $T_A$. (a) Comparison of `jaxmg.potrs` with `jax.scipy.linalg.cho_factor` + `jax.scipy.linalg.cho_solve` for a float32 matrix. The largest solvable problem is `N=524288`, which utilizes >1 TB of memory. (b) Comparison of `jaxmg.potri` with `jax.numpy.linalg.inv` for a complex128 matrix. (c) Comparison of `jaxmg.syevd` with `jax.numpy.linalg.eigh`.\label{fig:benchmark}](figures/jaxmg_benchmark.png){ width=100% }

[^1]: Random positive definite matrices give the same timings.

# AI usage disclosure

GitHub Copilot was used during software development for code exploration and debugging. Large language models were used to assist with language polishing.

# Acknowledgements

I want to thank Dennis Bollweg, Alex Chavin, Geraud Krawezik, Dylan Simon and Nils Wentzell for their help with developing the code. I also want to acknowledge the help of Ao Chen and Riccardo Rende with testing the code in applied settings.  
I am grateful to Simon Tartakovsky for his suggestions on the 1D cyclic algorithm. Finally, I want to thank
 Filippo Vincentini for his suggestions on code distribution. I acknowledge support from the Flatiron Institute. The Flatiron Institute is a division of the Simons Foundation. 


# References