# Continuous Cascaded Shadow Maps Research

Research project investigating a continuous alternative to traditional Cascaded Shadow Maps (CSM) using compute/software rasterization.

The project uses [LightweightVK](https://github.com/corporateshark/lightweightvk) as the rendering framework and experimentation platform.

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

Each cascade renders a separate orthographic shadow map with different world-space coverage and texel density.

This works well but introduces several drawbacks:

- Multiple shadow rendering passes
- Duplicate rendering across cascades
- Increased draw submission cost
- Visible cascade transitions
- Per-cascade tuning
- Additional memory overhead

CSM already describes the desired texel density distribution:

```text
near camera -> high density
far camera -> low density
```

but approximates it using a small number of discrete levels.

This project investigates replacing those discrete levels with a continuous representation, effectively creating "infinite cascades".

Removing discrete cascades may also reduce overlap and duplicate caster inclusion requirements present in traditional CSM implementations.

Potential advantages:

- One shadow pass
- One shadow representation
- No cascade transitions
- Reduced overlap and duplicate caster rendering
- Reduced draw submission overhead
- Potentially lower memory usage
- Enables continuous Nanite-style LOD behavior inside the shadow representation
- Enables async compute overlap with main view rendering

---

## Idea

Keep the same conceptual setup as CSM:

- Directional light
- Camera-frustum driven allocation
- Familiar split controls and tuning workflow

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

becomes a continuous density function instead of several discrete regions.

The intention is to preserve the existing CSM workflow and potentially expose nearly identical controls externally while changing only the internal representation.

Traditional CSM parameters such as:

```cpp
bias[cascade]
slope_bias[cascade]
filter_radius[cascade]
```

could potentially become:

```cpp
bias(depth)
slope_bias(depth)
filter_radius(depth)
```

allowing continuous behavior while preserving familiar tuning controls.

---

## Why Software Rasterization?

Previous warped shadow techniques such as Perspective Shadow Maps, Light Space Perspective Shadow Maps and Trapezoidal Shadow Maps were constrained by fixed function hardware rasterization.

Hardware rasterization assumes projected triangle edges remain linear. More aggressive nonlinear parameterizations may produce curved projections that are not naturally representable by traditional pipelines.

Compute/software rasterization removes this restriction and potentially allows exploration of a much larger space of shadow parameterizations.

Modern GPU software rasterization approaches such as Nanite have also shown that small triangles can be competitive with, and sometimes outperform, traditional hardware rasterization.

This work investigates whether software rasterization cost can be offset through a single shadow pass, reduced draw duplication and a more compact shadow representation.

---

## Progress

✅ Setup repository, LightweightVK and build system  
✅ Render a test scene (Bistro/Sponza or similar)  
✅ Implement baseline shadow mapping  
⬜ Extend baseline shadow mapping to CSM  
⬜ Implement continuous shadow parameterization using hardware rasterization  
⬜ Integrate software rasterization  
⬜ Compare against traditional CSM

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
