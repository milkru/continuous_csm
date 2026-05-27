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

Port of `3rdparty/lightweightvk/samples/Tiny_MeshLarge.cpp` (Bistro exterior scene), used as the rendering baseline before CSM work begins. Split across several files:

| File | Contents |
|---|---|
| `src/main.cpp` | All LVK globals, render passes, pipelines, `init`/`destroy`/`render`, GLFW callbacks, `main()` |
| `src/gui.h` / `src/gui.cpp` | `showTimeGPU()` stats overlay with custom ImDrawList sparklines; owns `GPUTimestamp` enum |
| `src/model.h` / `src/model.cpp` | Bistro OBJ loading, mesh cache, material/texture streaming via taskflow |
| `src/camera.h` | Flat `Camera` class — no includes, no `using`, all `glm::` prefixed |
| `src/shaders.h` | All GLSL shader source strings as `const char*` constants |
| `src/utils.h` | `ScopeExit_` / `SCOPE_EXIT`, `msign`, `packOctahedral16` (header-only) |

**Key dependency**: LightweightVK (`3rdparty/lightweightvk/`) — a bindless-only Vulkan 1.3+ wrapper. It is a git submodule. The public API lives in `3rdparty/lightweightvk/lvk/LVK.h`.

LightweightVK has its own `CLAUDE.md` at `3rdparty/lightweightvk/CLAUDE.md` — consult it for conventions and build details of the library itself (do not modify submodule files without good reason).

**Standard**: C++20.

## CMake / Link Targets

`LVK_WITH_SAMPLES` is forced **OFF**. We link only the public `LVKLibrary` target (not the sample helper `LVKVulkanApp`). Third-party dependencies are managed as our own git submodules under `3rdparty/`.

```cmake
set(LVK_WITH_SAMPLES OFF CACHE BOOL "" FORCE)

add_subdirectory(3rdparty/lightweightvk)
add_subdirectory(3rdparty/meshoptimizer)

target_link_libraries(continuous_csm PRIVATE LVKLibrary meshoptimizer)
target_include_directories(continuous_csm PRIVATE
    "${CMAKE_SOURCE_DIR}/3rdparty/fast_obj"
    "${CMAKE_SOURCE_DIR}/3rdparty/meshoptimizer/src"
    "${CMAKE_SOURCE_DIR}/3rdparty/stb"
    "${CMAKE_SOURCE_DIR}/3rdparty/taskflow")
```

Our own submodules (`.gitmodules`):
- `3rdparty/lightweightvk` — LVK core
- `3rdparty/meshoptimizer` — vertex cache / overdraw optimisation
- `3rdparty/fast_obj` — header-only OBJ parser (`FAST_OBJ_IMPLEMENTATION` defined in `model.cpp`)
- `3rdparty/stb` — image load/resize (`STB_IMAGE_IMPLEMENTATION`, `STB_IMAGE_RESIZE_IMPLEMENTATION` defined in `model.cpp`)
- `3rdparty/taskflow` — async texture loading (header-only)

`LVKLibrary` exposes these **PUBLIC** includes that propagate to us automatically:
- GLFW headers
- `deps/src/imgui` — ImGui headers (use `<imgui/imgui.h>` if needed directly)

## Code Style

- **Braces**: Allman — opening brace on its own line, always, including single-statement blocks.
- **Indentation**: tabs.
- **Integer types**: `uint32_t`, `int32_t`, `uint64_t`, etc. Never `int`, `unsigned`, `long`.
- **File names**: lowercase snake_case (`camera.h`, `shadow_map.cpp`).
- **No `using` declarations** anywhere — headers or `.cpp`. Always write `glm::vec3`, `glm::mat4`, etc. in full.
- **No includes in project headers** — `.h` files we own must not contain `#include`. The includer is responsible for pulling in all dependencies before the header.
- **Header include order**: standard library → third-party → project headers (always last).

## LightweightVK Usage Notes

- Bindless-only: no descriptor sets per draw; use buffer device addresses and descriptor indexing.
- `lvk::LVKwindow*` + `lvk::createVulkanContextWithSwapchain()` is the entry point (see `main.cpp`).
- `lvk::ImGuiRenderer` wraps ImGui init/frame/submit — created in `init()`, owned by `imgui_`.
- For compute/software rasterization work (the core of this research), look at how LVK dispatches compute shaders; hardware rasterization pipelines are available but the research direction favors compute.
