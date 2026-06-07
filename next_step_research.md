# Continuous CSM Notes

## Goal

The final goal is to build a continuous replacement for CSM while keeping the shadow map aligned with the light source, similar to how regular CSM works.

However, this document investigates a simpler first step:

```text
camera/frustum aligned shadow map
+
continuous logarithmic warp
```

The reason is that it is easier to reason about and closely follows the logic behind logarithmic CSM splits.

The idea is to first understand the continuous density distribution problem, then later investigate how to apply similar ideas to a light-aligned shadow map.

---

## CSM Background

CSM splits the camera frustum into multiple depth ranges.

Pure logarithmic split:

```cpp
split[i] = near_z * pow(far_z / near_z, float(i) / cascade_count);
```

The important observation:

```cpp
depth = near_z * pow(far_z / near_z, t);
```

can be inverted:

```cpp
t = log(depth / near_z) / log(far_z / near_z);
```

This converts a depth value into a continuous cascade index.

Conceptually:

```text
CSM:
    cascade index -> depth

Continuous warp:
    depth -> continuous cascade index
```

---

## Continuous Warp Idea

Assumptions:

```text
shadow X = frustum X
shadow Y = frustum depth Z
```

Warp:

```cpp
u = x / (z * tan_half_fov_x);
v = log(z / near_z) / log(far_z / near_z);
```

where:

```cpp
z = -view_pos.z;
```

This is effectively the inverse of the logarithmic split formula applied continuously.

Expected result:

```text
more texels near camera
fewer texels far away
no cascade transitions
```

---

## Hardware Rasterization Version

Assume hardware rasterization works correctly.

Vertex shader:

```cpp
float z = -view_pos.z;

float u =
    view_pos.x /
    (z * tan_half_fov_x);

float v =
    log(z / near_z) /
    log(far_z / near_z);

gl_Position =
{
    u,
    v * 2.0f - 1.0f,
    depth,
    1.0f
};
```

Expected behaviour:

```text
continuous logarithmic density
similar behaviour to infinite logarithmic CSM
```

This version is useful for validating the warp itself.

---

## Why Hardware Rasterization Breaks

Hardware rasterization assumes:

```text
3D line -> 2D line
```

after projection.

For this warp:

```cpp
u = x / z
v = log(z)
```

triangle edges become curves.

Example:

```text
Original:

      *
     / \
    /   \
   *-----*

Warped:

      *
    )   (
   )     (
  *-------*
```

A normal GPU rasterizer would rasterize the straight triangle defined by the transformed vertices, which is incorrect.

Because of this, hardware rasterization can only be used as a prototype.

A correct solution requires curved triangle rasterization.

---

## Inverse Warp

Given a warped point:

```cpp
u
v
```

recover original coordinates:

```cpp
z = near_z * pow(far_z / near_z, v);
x = u * z * tan_half_fov_x;
```

This inverse is simple and analytic.

---

## Software Rasterization Version

Instead of deriving curved edge equations, inverse-warp the sample point and use the original triangle edge equations.

Key observation:

```text
Warped triangle coverage

is equivalent to

Inverse-warped pixel inside original triangle
```

---

## Regular Rasterization

```cpp
bool inside_triangle(P2 p, P2 v0, P2 v1, P2 v2)
{
    float e0 = edge(v0, v1, p);
    float e1 = edge(v1, v2, p);
    float e2 = edge(v2, v0, p);

    return same_sign(e0, e1, e2);
}
```

---

## Curved Rasterization

```cpp
P2 inverse_warp(float2 warped)
{
    float z =
        near_z *
        pow(far_z / near_z, warped.y);

    float x =
        warped.x *
        z *
        tan_half_fov_x;

    return { x, z };
}

bool inside_curved_triangle(
    float2 pixel_warped,
    P2 v0,
    P2 v1,
    P2 v2)
{
    P2 pixel =
        inverse_warp(pixel_warped);

    float e0 = edge(v0, v1, pixel);
    float e1 = edge(v1, v2, pixel);
    float e2 = edge(v2, v0, pixel);

    return same_sign(e0, e1, e2);
}
```

The edge tests are identical.

The only extra work is:

```cpp
pixel = inverse_warp(pixel);
```

before the edge tests.

---

## Cost Comparison

Regular:

```cpp
edge0(pixel)
edge1(pixel)
edge2(pixel)
```

Warped:

```cpp
pixel = inverse_warp(pixel);

edge0(pixel)
edge1(pixel)
edge2(pixel)
```

Coverage testing stays almost identical.

---

## Bounding Boxes

Still required.

Without a bounding box:

```cpp
for every pixel in shadow map
```

With a bounding box:

```cpp
for every pixel in triangle bbox
```

Huge performance difference.

For this warp:

```cpp
u = x / z
v = log(z)
```

both dimensions are monotonic along an edge.

Initial assumption:

```cpp
bbox = min/max(warped vertices)
```

may already be sufficient.

Needs verification.

---

## Triangle Edge Shape

Original edge:

```cpp
x(t) = x0 + t * dx
z(t) = z0 + t * dz
```

Warped edge:

```cpp
u(t) = x(t) / z(t)
v(t) = log(z(t))
```

After eliminating t:

```cpp
u = A + B * exp(-v)
```

So edges become simple exponential curves.

---

## Depth Terminology

Linear depth:

```cpp
linear_depth = -view_pos.z;
```

Not:

```cpp
length(view_pos);
```

Examples:

```cpp
viewPos = (10, 0, -10)

linear_depth = 10
distance = 14.14
```

CSM uses linear depth.

---

## Cascade Visualization

CSM cascade selection:

```cpp
if (depth < split0)
    cascade = 0;
else if (depth < split1)
    cascade = 1;
...
```

where:

```cpp
depth = -viewPos.z;
```

Cascade boundaries are planes perpendicular to camera forward direction.

Visualized on screen they appear as straight horizontal bands.

---

## Shadow Depth

For the frustum-aligned version discussed in this document:

```cpp
shadow_depth = z;
```

or:

```cpp
shadow_depth = normalize(z);
```

Inverse-warp already gives the depth required for storage.

For a future light-aligned version:

```cpp
shadow_depth = dot(world_pos, light_dir);
```

and full position reconstruction would be required.

---

## Axis Projectivity

The warp is separable into two axes:

```text
lateral = x / depth   (perspective, x/z)
depth   = log(depth)
```

Lateral is projective: it preserves straight lines, and because the receiver
evaluates the warp analytically per fragment, it yields uniform horizontal texel
density on screen.

Depth is non-projective: `log` cannot be expressed as a homogeneous divide, so it
curves straight lines.

Hardware limit:

```text
one homogeneous w  ->  only one axis can be projective
```

Routing the lateral through `w` makes it exact, but the depth axis must then share
that `w` and is no longer exact (it gets perspective-interpolated, not log). So in
hardware at most one axis is straight; the other curves.

Current prototype does the lateral divide by hand (`w = 1`), so both axes curve in
the stored map. On-screen density stays uniform regardless (receiver is analytic);
only the caster's stored edges curve, causing minor edge artifacts.

Software rasterization removes the conflict entirely:

```text
per-pixel inverse_warp  ->  exact coverage and depth, both axes
```

No rasterizer interpolation means exact perspective lateral and exact log depth at
once, with no curving and no shared-w trade. Both properties come for free.

### Rasterization cost

The straight lateral also shrinks the curved-rasterization cost (the gap between
regular and curved software raster). Iterate the shadow map by rows of constant
depth:

```text
per row:     one nonlinear depth inverse (pow)
within row:  depth is constant -> lateral inverse x = u * width(depth) is linear in u
             -> regular incremental edge stepping, no per-pixel inverse_warp
```

So the inner loop is an ordinary straight-edge scanline rasterizer; the only curved
cost is one solve per scanline.

```text
both axes non-projective:  inverse_warp per pixel   (~N^2 solves)
projective lateral:        depth inverse per row     (~N solves) + regular inner loop
```

Projectivity is what unlocks this: linear u <-> x within a row makes the edge
functions linear, so they step incrementally. A non-projective lateral (e.g.
u = sqrt(x / width)) is nonlinear in u even within a row, forcing per-pixel inverse
warp again.

---

## Future Investigation

The current approach uses:

```text
camera aligned shadow map
```

The final goal is likely:

```text
light aligned shadow map
+
continuous density warp
```

which would be much closer to how CSM actually works.

The main challenge is that a single light-space location can correspond to many different camera depths, so the logarithmic density distribution is no longer uniquely defined.

This is likely the hardest part of a true continuous CSM replacement.

---

## Main Takeaway

The hardware version provides a simple proof of concept for a continuous logarithmic CSM-like warp.

The software version keeps the original triangle edge tests and only adds:

```cpp
pixel = inverse_warp(pixel);
```

before rasterization.

This makes the curved rasterization problem much simpler than expected.
