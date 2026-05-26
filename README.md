# Continuous Cascaded Shadow Maps Research

Research project investigating a continuous alternative to traditional Cascaded Shadow Maps (CSM) using compute/software rasterization.

The goal is to preserve the familiar behavior and controls of CSM while replacing its discrete cascade representation with a continuous shadow parameterization.

---

## Motivation

Traditional CSM improves shadow quality by splitting the camera frustum into multiple depth intervals:

```text
Cascade 0: near
Cascade 1
Cascade 2
Cascade 3: far
```

Each cascade renders a separate orthographic shadow map with different world-space coverage and effective texel density.

This works well but introduces several drawbacks:

- Multiple shadow rendering passes
- Duplicate rendering of nearby objects across cascades
- Increased draw submission cost
- Cascade transition artifacts
- Per-cascade stabilization logic
- Per-cascade tuning (bias, slope bias, filtering)
- Additional memory overhead

CSM already defines the desired texel density distribution:

```text
near camera -> high density
far camera -> low density
```

but approximates it using several discrete levels.

This project investigates replacing those discrete steps with a continuous representation:

```text
continuous_density(depth)
```

or effectively:

```text
"infinite cascades"
```

Removing discrete cascades may also reduce overlap and duplicate caster inclusion requirements present in traditional CSM implementations.

Potential advantages:

- One shadow pass
- One shadow representation
- No cascade transitions
- No cascade overlap
- Reduced duplicate rendering
- Potentially lower memory usage
- Enables Nanite-style LOD behavior directly inside the shadow representation

Traditional CSM approximates shadow LOD with a small number of discrete levels. This work investigates whether shadow resolution can instead become a continuous function.

---

## Idea

Keep the same conceptual setup as CSM:

- Directional light
- Camera-frustum driven allocation
- Split controls
- Familiar tuning workflow

Internally replace:

```text
4 orthographic cascades
```

with:

```text
single continuous shadow parameterization
```

Conceptually:

```text
0-10m      high density
10-40m     medium density
40-150m    low density
150-500m   very low density
```

becomes:

```text
continuous_density(depth)
```

The idea is to preserve existing CSM workflows and potentially even expose identical controls externally while changing only the internal representation.

---

## Why Software Rasterization?

Previous shadow warping approaches were constrained by fixed-function rasterization and projective transforms.

Compute/software rasterization removes this restriction and allows exploration of nonlinear shadow-space parameterizations not naturally representable by traditional pipelines.

---

## Continuous Parameters

Traditional CSM often exposes per-cascade settings:

```cpp
bias[cascade]
slope_bias[cascade]
filter_radius[cascade]
```

Continuous CSM could replace these with:

```cpp
bias(depth)
slope_bias(depth)
filter_radius(depth)
```

allowing continuous behavior while preserving familiar tuning controls.

---

## Research Questions

### Projection

Can a nonlinear shadow parameterization preserve directional light visibility, stable texel distribution, robustness, smooth transitions and practical filtering?

### Rasterization

Can software rasterization enable mappings not feasible with fixed-function pipelines?

### Performance

Modern GPU software rasterization approaches such as Nanite have shown that small triangles can be competitive with, and sometimes outperform, traditional hardware rasterization.

This work investigates whether the cost of software rasterization can be offset by reducing shadow rendering complexity through a single shadow pass, reduced draw duplication and potentially lower memory usage.

---

## References

- [LearnOpenGL Cascaded Shadow Maps](https://learnopengl.com/Guest-Articles/2021/CSM)
- [LightweightVK](https://github.com/corporateshark/lightweightvk)
- [3D Graphics Rendering Cookbook](https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition)
- [Sascha Willems Cascaded Shadow Maps](https://github.com/SaschaWillems/Vulkan/tree/master/examples/shadowmappingcascade)
- [Perspective Shadow Maps](https://www-sop.inria.fr/reves/Basilic/2002/SD02)
- [Light Space Perspective Shadow Maps](https://www.cg.tuwien.ac.at/research/vr/lispsm/)
- [Trapezoidal Shadow Maps](https://www.comp.nus.edu.sg/~tants/tsm.html)
- [Nanite SIGGRAPH Presentation](https://advances.realtimerendering.com/s2021/index.html#_oci4k4s9u6rp)
