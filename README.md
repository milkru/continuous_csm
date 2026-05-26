# Continuous Cascaded Shadow Maps Research

Research project investigating a continuous alternative to traditional
Cascaded Shadow Maps (CSM) using software/compute rasterization.

The goal is not to invent a completely new shadowing technique but to
preserve the external behavior and controls of CSM while replacing its
discrete cascade representation with a continuous parameterization.

------------------------------------------------------------------------

## Motivation

Traditional CSM improves shadow quality by splitting the camera frustum
into multiple depth intervals:

``` text
Cascade 0: near
Cascade 1
Cascade 2
Cascade 3: far
```

Each cascade renders a separate orthographic shadow map with different
world-space coverage and effective texel density.

Problems:

-   Multiple shadow rendering passes
-   Duplicate rendering of nearby objects across cascades
-   Increased draw submission cost
-   Cascade transition artifacts
-   Per-cascade stabilization logic
-   Per-cascade tuning (bias, slope bias, filtering)
-   Additional memory overhead

Observation:

``` text
near camera -> high density
far camera -> low density
```

CSM already defines the desired texel distribution, but approximates it
with discrete steps.

------------------------------------------------------------------------

## Idea

Keep the same external setup as CSM:

-   Directional light
-   Camera-frustum driven allocation
-   Split controls
-   Familiar tuning workflow

Internally replace:

``` text
4 orthographic cascades
```

with:

``` text
single continuous shadow parameterization
```

Conceptually:

``` text
0-10m      high density
10-40m     medium density
40-150m    low density
150-500m   very low density
```

becomes:

``` text
continuous_density(depth)
```

or effectively:

``` text
"infinite cascades"
```

Goals:

-   One shadow pass
-   One shadow representation
-   No cascade transitions
-   No repeated rendering across cascades
-   Continuous shadow LOD

------------------------------------------------------------------------

## Why Software Rasterization?

Previous shadow warping approaches had to preserve:

``` text
triangle
-> projective transform
-> triangle
```

Software/compute rasterization potentially allows:

``` text
triangle
-> nonlinear transform
-> curved boundaries
```

This enables exploration of shadow-space mappings not constrained by
fixed-function hardware assumptions.

Assumptions:

-   Depth-only rendering
-   Tile binning
-   Small triangles
-   Aggressive culling
-   Compute rasterization

This is not intended to become a full software renderer. Only shadow
depth generation is relevant.

------------------------------------------------------------------------

## Continuous Parameters

Traditional CSM:

``` cpp
bias[cascade]
slope_bias[cascade]
filter_radius[cascade]
```

Continuous CSM:

``` cpp
bias(depth)
slope_bias(depth)
filter_radius(depth)
density(depth)
```

------------------------------------------------------------------------

## Research Questions

### Projection

Can a nonlinear shadow parameterization preserve:

-   Directional light visibility
-   Stable texel distribution
-   Robustness
-   Smooth transitions
-   Practical filtering

### Rasterization

Can software rasterization enable mappings not feasible with
fixed-function pipelines?

### Performance

Can:

-   One shadow pass
-   Fewer draw calls
-   No cascade duplication
-   Smaller memory footprint

offset software rasterization cost?

------------------------------------------------------------------------

## Roadmap

### Phase 1

-   LightweightVK
-   Cookbook shadow mapping sample
-   Traditional CSM baseline

### Phase 2

Replace:

``` text
discrete split regions
```

with:

``` text
continuous density(depth)
```

Investigate:

-   Logarithmic mappings
-   Integrated split distributions
-   Nonlinear shadow-space warps

### Phase 3

Compute/software shadow rasterization:

-   Tile binning
-   Depth-only rasterization
-   Custom shadow parameterization

### Phase 4

Compare against CSM:

Visual: - Quality - Stability - Transitions - Aliasing

Performance: - GPU time - Draw calls - Memory - Raster cost

------------------------------------------------------------------------

## References

LearnOpenGL CSM\
https://learnopengl.com/Guest-Articles/2021/CSM

LightweightVK\
https://github.com/corporateshark/lightweightvk

3D Graphics Rendering Cookbook\
https://github.com/PacktPublishing/3D-Graphics-Rendering-Cookbook-Second-Edition

Sascha Willems Vulkan CSM\
https://github.com/SaschaWillems/Vulkan/tree/master/examples/shadowmappingcascade

Perspective Shadow Maps\
https://www-sop.inria.fr/reves/Basilic/2002/SD02

Light Space Perspective Shadow Maps\
https://www.cg.tuwien.ac.at/research/vr/lispsm/

Trapezoidal Shadow Maps\
https://www.comp.nus.edu.sg/\~tants/tsm.html

Nanite\
https://advances.realtimerendering.com/s2021/index.html#\_oci4k4s9u6rp

RetroWarp\
https://github.com/Themaister/RetroWarp
