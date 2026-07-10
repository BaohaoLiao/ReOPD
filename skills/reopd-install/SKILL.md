---
name: reopd-install
description: Set up the full ReOPD training and evaluation environment. Use this skill whenever the user wants to install, set up, or troubleshoot the ReOPD environment — including the slime training stack, the ReTool (math with Python) task, the Search-R1 task, the separate retriever conda environment, or faiss GPU builds. Also use it for GPU-specific setup questions (A100 vs H100), import errors after installation, or "environment not working" issues in this repo.
---

# ReOPD Installation

ReOPD has one base training environment plus task-specific extras. Install in
this order — the base environment first, then only the extras for the task(s)
the user needs. Ask which GPUs they have (A100 vs H100/H800) early: it changes
both the slime image and the faiss install path.

## 1. Base environment (slime)

The training stack (PyTorch + CUDA, Ray, SGLang, Megatron-LM) comes from
slime's environment, not from this repo. Follow slime's official quick-start:
https://github.com/THUDM/slime/blob/main/docs/en/get_started/quick_start.md

GPU note — slime's default images target H100/H800 (SM90):

- **H100/H800**: use the recommended Docker image from the quick-start as-is.
- **A100 (SM80)**: follow https://github.com/THUDM/slime/pull/1832, which adds
  an A100 patch set (`PATCH_VERSION=v0.5.9.a100`), an offline-friendly conda
  build (`build_conda.a100.sh`), and `docker/Dockerfile.a100`.

Then install ReOPD on top, replacing the slime checkout with our fork
(the fork is required — it carries OPD support and stability fixes):

```bash
git clone https://github.com/BaohaoLiao/ReOPD.git
cd ReOPD
git submodule update --init --recursive
pip install -r requirements.txt
pip install -e third_party/slime --no-deps
```

Verify:

```bash
python -c "import slime; import torch; print(torch.__version__, torch.cuda.is_available())"
python -c "import math_verify, latex2sympy2_extended, pylatexenc; print('math grading OK')"
```

## 2. Math with Python (ReTool) task

No extra installation. The Python tool sandbox runs inside the training
environment and its dependencies (`jupyter_client`, `ipykernel`, `psutil`,
`sympy`, ...) are already covered by `requirements.txt`.

## 3. Search (Search-R1) task

Two parts: Search-R1 dependencies in the training environment, and a separate
conda environment for the local dense retriever.

### 3a. Training environment extras

```bash
pip install chardet tensordict
git clone https://github.com/PeterGriffinJin/Search-R1.git && cd Search-R1
pip install -e . --no-deps
```

### 3b. Retriever environment (separate on purpose)

The retriever needs torch 2.4 and an older transformers pin, which conflict
with the training stack — that is why it lives in its own conda env. Do not
install these into the training environment.

```bash
conda create -n retriever python=3.10 -y
conda activate retriever
conda install pytorch==2.4.0 pytorch-cuda=12.1 -c pytorch -c nvidia -y
# transformers must stay <4.47: newer versions call torch APIs missing in 2.4
# ("infer_schema(...) Parameter input has unsupported type torch.Tensor")
pip install "transformers==4.46.3" datasets pyserini huggingface_hub
pip install uvicorn fastapi
# conda's torchvision often mismatches torch and is unused by the server
pip uninstall -y torchvision torchaudio
```

### 3c. faiss with GPU support

Pick by GPU:

- **A100 (SM80)**: the PyPI wheel works: `pip install faiss-gpu-cu12`
- **H100 (SM90)**: the PyPI wheel ships kernels only for SM 7.0–8.9 and
  crashes with `CUDA error 209 no kernel image is available`. Build faiss
  v1.9.0 from source:

```bash
conda install -y -c conda-forge cmake "swig=4.2.*" mkl mkl-devel

git clone https://github.com/facebookresearch/faiss.git
cd faiss
git checkout v1.9.0   # v1.8.0's swig file is incompatible with modern swig

# 80=A100, 90=H100; for Blackwell (B100/B200) add 100
cmake -B build . \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON \
  -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF \
  -DFAISS_OPT_LEVEL=avx2 \
  -DCMAKE_CUDA_ARCHITECTURES="80;90" \
  -DPython_EXECUTABLE=$(which python) \
  -DSWIG_EXECUTABLE=$(which swig)

make -C build -j$(nproc) faiss swigfaiss
cd build/faiss/python && pip install .
```

Verify (must print `True` for GPU symbols):

```bash
python -c "import faiss; print(faiss.__file__, faiss.__version__, hasattr(faiss,'GpuMultipleClonerOptions'))"
```

Build gotchas:

- `swig=4.2.*` is required — v1.9.0's swig file compiles with neither system
  swig 3.x (`SWIGTYPE_p_unsigned_long_long was not declared`) nor swig 4.4.x.
- If a system `swig` shadows the conda one, force it with
  `-DSWIG_EXECUTABLE=$(which swig)` or `export PATH=$CONDA_PREFIX/bin:$PATH`.

CPU fallback (fine for eval — GPU encoder + CPU index is fast enough):

```bash
pip uninstall -y faiss faiss-gpu faiss-gpu-cu12 2>/dev/null
pip install faiss-cpu
# then drop --faiss_gpu from the retrieval_server.py launch command
```

## Common failures

| Symptom | Cause / fix |
| --- | --- |
| `CUDA error 209 no kernel image` from faiss | PyPI wheel on H100 — build from source (3c) |
| `infer_schema(...) unsupported type torch.Tensor` | transformers too new for torch 2.4 — pin `transformers==4.46.3` |
| `operator torchvision::nms does not exist` | conda torchvision mismatches torch — `pip uninstall torchvision torchaudio` |
| `ModuleNotFoundError: math_verify` during math eval | `pip install -r requirements.txt` in the training env |
| slime import errors / missing OPD args | using upstream slime — install the fork: `pip install -e third_party/slime --no-deps` |
