# Installation

This repository is a downstream development branch of JAXMg. It is not the
released PyPI package. Installing the package name `jaxmg` from PyPI will install
the upstream release, not the branch-specific log determinant support in this
repository.

This branch adds native `log|A|` output to the multi-GPU `potrs` workflow via
`return_logdet=True`.

There are two supported installation paths for this downstream branch.

## 1. Build from source

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

## 2. Prebuilt wheels

Use this path when a wheel is provided for an environment matching your Python,
CUDA, and system architecture. The wheel already contains the native shared
libraries, so users do not need to run CMake locally. If you need to run CMake,
use the source build path above instead.

The included wheel was built on CSD3 for Linux x86_64, CPython 3.11, CUDA 12.1,
cuDNN 8.9, and `jax[cuda12-local]==0.10.1`. It is intended for CSD3 and systems
with a compatible software stack.

To install the included wheel, clone this repository and install from the
relative wheel path:

```bash
git clone https://github.com/JacobTutt/jaxmg.git
cd jaxmg
pip install "wheels/csd3/jaxmg-0.0.7-cp311-cp311-linux_x86_64.whl[csd3]"
```

The `csd3` extra installs the JAX runtime used for this wheel:
`jax[cuda12-local]==0.10.1`.

See `CONTRIBUTING.md` for more build details.
