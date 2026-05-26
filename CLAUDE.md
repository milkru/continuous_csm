# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Research project investigating **Continuous Cascaded Shadow Maps (CSM)** — replacing traditional discrete cascade shadow maps with a single continuous shadow parameterization using compute/software rasterization on Vulkan via [LightweightVK](https://github.com/corporateshark/lightweightvk).

The core idea: instead of 4 orthographic cascades, use one shadow pass with `continuous_density(depth)` that internally approximates "infinite cascades". See README.md for the full research motivation and open questions.

## Build

**Prerequisites**: CMake 3.22+, Python 3, Visual Studio 2022, Vulkan SDK

```bat
# Generate VS solution (run from repo root)
cmake -S . -B build

# Or use the helper script
GenerateProjectFiles.bat
```

CMake automatically runs `deploy_deps.py` and `deploy_content.py` from the LightweightVK submodule during configuration — these download Vulkan headers, GLFW, GLM, ImGui, and sample assets. This requires internet access and takes time on first run.

```bat
# Build from command line after generation
cmake --build build --config Release --parallel
```

The generated `continuous_csm.sln` opens directly in Visual Studio 2022. The startup project is set to `continuous_csm`.

## Architecture

**Single source file** (for now): `src/main.cpp` — implementation has not started yet.

**Key dependency**: LightweightVK (`3rdparty/lightweightvk/`) — a bindless-only Vulkan 1.3+ wrapper. It is a git submodule. The public API lives in `3rdparty/lightweightvk/lvk/LVK.h`. Link target is `LVKLibrary`.

LightweightVK has its own `CLAUDE.md` at `3rdparty/lightweightvk/CLAUDE.md` — consult it for conventions and build details of the library itself (do not modify submodule files without good reason).

**Standard**: C++20.

## LightweightVK Usage Notes

- Bindless-only: no descriptor sets per draw; use buffer device addresses and descriptor indexing.
- Context creation follows the pattern in LightweightVK samples (`3rdparty/lightweightvk/samples/`).
- For compute/software rasterization work (the core of this research), look at how LVK dispatches compute shaders; hardware rasterization pipelines are available but the research direction favors compute.
