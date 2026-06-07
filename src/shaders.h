#pragma once

// Shared continuous-shadow warp. The shadow map is camera-aligned, so near->far is the
// texture's vertical (+y) axis and lateral is x. That makes the warp separable -- no
// orientation vectors needed:
//   y: logarithmic depth distribution = log(z/near)/log(far/near) (texel density
//      ~ 1/depth, the same falloff as CSM's log splits). tlin is linear in depth, so
//      z/near = 1 + tlin*(ratio-1) with ratio = far/near.
//   x: divide by the frustum half-width at this depth (the perspective x/z fill, so the
//      narrow near slice stretches wide)
// It's a pure function of light-NDC position -> a bijection -> occlusion is preserved.
// Params come straight from the frustum footprint: aNear/aFar = near/far center y,
// wNear/wFar = x half-widths at near/far. Applied identically in caster VS and lookup.
#define WARP_LIGHT_NDC_GLSL R"(
vec2 warpLightNDC(vec2 p, float aNear, float aFar, float wNear, float wFar, float ratio) {
  float tlin = clamp((p.y - aNear) / (aFar - aNear), 0.0, 1.0); // 0 near .. 1 far
  float oy   = log(1.0 + tlin * (ratio - 1.0)) / log(ratio) * 2.0 - 1.0; // logarithmic depth
  float ox   = p.x / max(mix(wNear, wFar, tlin), 1e-4);                  // lateral fill
  return vec2(ox, oy);
}
)"

const char* kCodeVS = R"(
layout (location=0) in vec3 pos;
layout (location=1) in vec2 uv;
layout (location=2) in uint normal;
layout (location=3) in uint mtlIndex;

struct Material {
  vec4 ambient;
  vec4 diffuse;
  int texAmbient;
  int texDiffuse;
  int texAlpha;
  int padding;
};

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 cascadeLightMatrices[4];
  vec4 cascadeSplitDepths;
  uint texShadow[4];
  uint sampler0;
  uint samplerShadow0;
  uint texShadowContinuous;
  uint shadowMode;
  vec4 ambientColor;
  vec4 lightDir;
  mat4 continuousLightViewProj;
  vec4 continuousWarpParams;  // x = aNear, y = aFar, z = wNear, w = wFar
  float continuousWarpRatio;  // far / near (logarithmic depth)
  uint pcfEnabled;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
  mat4 normal;
};

layout(std430, buffer_reference) readonly buffer Materials {
  Material mtl[];
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
  Materials materials;
} pc;

struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 worldPos; // .xyz = world-space position, .w = view-space Z
};
layout (location=0) out PerVertex vtx;
layout (location=5) flat out Material mtl;

vec2 unpackSnorm2x8(uint d) {
  return vec2(uvec2(d, d >> 8) & 255u) / 127.5 - 1.0;
}
vec3 unpackOctahedral16(uint data) {
  vec2 v = unpackSnorm2x8(data);
  vec3 n = vec3(v, 1.0 - abs(v.x) - abs(v.y));
  float t = max(-n.z, 0.0);
  n.x += (n.x > 0.0) ? -t : t;
  n.y += (n.y > 0.0) ? -t : t;
  return normalize(n);
}

void main() {
  mat4 proj  = pc.perFrame.proj;
  mat4 view  = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  mtl = pc.materials.mtl[mtlIndex];
  vec4 worldPos = model * vec4(pos, 1.0);
  gl_Position = proj * view * worldPos;
  vtx.normal   = normalize(mat3(pc.perObject.normal) * unpackOctahedral16(normal));
  vtx.uv       = uv;
  vtx.worldPos = vec4(worldPos.xyz, (view * worldPos).z);
}
)";

const char* kCodeVS_Wireframe = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
} pc;

void main() {
  mat4 proj = pc.perFrame.proj;
  mat4 view = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
}
)";

const char* kCodeFS_Wireframe = R"(
layout (location=0) out vec4 out_FragColor;

void main() {
  out_FragColor = vec4(1.0);
};
)";

const char* kCodeFS = R"(
layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
  mat4 cascadeLightMatrices[4];
  vec4 cascadeSplitDepths;
  uint texShadow[4];
  uint sampler0;
  uint samplerShadow0;
  uint texShadowContinuous;
  uint shadowMode;
  vec4 ambientColor;
  vec4 lightDir;
  mat4 continuousLightViewProj;
  vec4 continuousWarpParams;  // x = aNear, y = aFar, z = wNear, w = wFar
  float continuousWarpRatio;  // far / near (logarithmic depth)
  uint pcfEnabled;
  uint shadowChecker;  // 1 = overlay shadow-texel checkerboard
  uint checkerTexels;  // shadow-map texels per checkerboard square
};

struct Material {
  vec4 ambient;
  vec4 diffuse;
  int texAmbient;
  int texDiffuse;
  int texAlpha;
  int padding;
};

struct PerVertex {
  vec3 normal;
  vec2 uv;
  vec4 worldPos;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
} pc;

layout (location=0) in PerVertex vtx;
layout (location=5) flat in Material mtl;
layout (location=0) out vec4 out_FragColor;

layout (constant_id = 0) const bool bDrawNormals = false;

float PCF3(uint texId, vec3 uvw) {
  float size = 1.0 / textureBindlessSize2D(texId).x;
  float shadow = 0.0;
  for (int v=-1; v<=+1; v++)
    for (int u=-1; u<=+1; u++)
      shadow += textureBindless2DShadow(texId, pc.perFrame.samplerShadow0, uvw + size * vec3(u, v, 0));
  return shadow / 9.0;
}
)" WARP_LIGHT_NDC_GLSL R"(
float shadow(vec3 worldPos, float viewZ, out vec3 checkerTint) {
  uint texId;
  vec4 sc;
  checkerTint = vec3(1.0);         // multiplied onto the lit scene (vec3(1) = no tint)
  if (pc.perFrame.shadowMode == 1u) {
    // Continuous: single light-space map fit to the frustum, with the depth-axis
    // log warp applied (same function used by the continuous shadow VS).
    vec4 lp = pc.perFrame.continuousLightViewProj * vec4(worldPos, 1.0);
    vec2 warped = warpLightNDC(lp.xy / lp.w,
        pc.perFrame.continuousWarpParams.x, pc.perFrame.continuousWarpParams.y,
        pc.perFrame.continuousWarpParams.z, pc.perFrame.continuousWarpParams.w,
        pc.perFrame.continuousWarpRatio);
    sc = vec4(warped * 0.5 + 0.5, lp.z / lp.w, 1.0);
    texId = pc.perFrame.texShadowContinuous;
  } else {
    // Discrete CSM: pick the cascade by view-space depth.
    uint cascadeIndex = 0;
    for (uint i = 0; i < 3; i++) {
      if (viewZ < pc.perFrame.cascadeSplitDepths[i])
        cascadeIndex = i + 1;
    }
    sc = pc.perFrame.cascadeLightMatrices[cascadeIndex] * vec4(worldPos, 1.0);
    texId = pc.perFrame.texShadow[cascadeIndex];
  }
  sc.xyz /= sc.w;
  // Acne is handled by hardware slope-scaled depth bias baked into the shadow map
  // during the depth pass, so no constant bias is needed here.
  vec3 uvw = vec3(sc.x, 1.0 - sc.y, sc.z);
  // Shadow-texel checkerboard debug overlay. The checker pattern is sized in shadow-map
  // texels (each square spans `checkerTexels` texels) and tinted by the world-space size
  // of a shadow texel on a green(fine)->red(coarse) scale. Discrete CSM has a constant
  // texel size per cascade -> one colour per cascade; the continuous map's log warp
  // varies it per fragment -> a smooth gradient on the same colour key.
  if (pc.perFrame.shadowChecker == 1u) {
    float texSize = float(textureBindlessSize2D(texId).x);
    // World-space size of one shadow texel at this fragment, recovered from screen-space
    // derivatives: (world units / pixel) / (shadow texels / pixel) = world units / texel.
    // The camera projection cancels, so it's distance-independent and works the same for
    // the discrete cascades (near-constant per cascade) and the continuous warp (per pixel).
    float worldPerPixel = length(fwidth(worldPos));
    float texelPerPixel = length(fwidth(uvw.xy * texSize));
    float worldTexel = worldPerPixel / max(texelPerPixel, 1e-6);
    // Green (fine) -> red (coarse) on a log2 scale. Bounds are in world units per texel and
    // are scene-scale dependent: tune these if the scene skews all-green or all-red.
    const float kGreenWorldTexel = 0.015;  // <= this reads green
    const float kRedWorldTexel   = 0.2;    // >= this reads red
    float t = clamp((log2(worldTexel) - log2(kGreenWorldTexel))
                    / (log2(kRedWorldTexel) - log2(kGreenWorldTexel)), 0.0, 1.0);
    // green -> yellow -> red, kept at full brightness through the middle for legibility.
    vec3 color = (t < 0.5) ? mix(vec3(0.0, 1.0, 0.0), vec3(1.0, 1.0, 0.0), t * 2.0)
                           : mix(vec3(1.0, 1.0, 0.0), vec3(1.0, 0.0, 0.0), t * 2.0 - 1.0);
    vec2 cell = floor(uvw.xy * texSize / float(pc.perFrame.checkerTexels));
    float dim = mod(cell.x + cell.y, 2.0) < 1.0 ? 1.0 : 0.5;
    checkerTint = color * dim;
  }
  float shadowSample = (pc.perFrame.pcfEnabled == 1u)
    ? PCF3(texId, uvw)
    : textureBindless2DShadow(texId, pc.perFrame.samplerShadow0, uvw);
  return mix(0.3, 1.0, shadowSample);
}

void main() {
  vec4 alpha = textureBindless2D(mtl.texAlpha, pc.perFrame.sampler0, vtx.uv);
  if (mtl.texAlpha > 0 && alpha.r < 0.5)
    discard;
  vec4 Ka = mtl.ambient * textureBindless2D(mtl.texAmbient, pc.perFrame.sampler0, vtx.uv);
  vec4 Kd = mtl.diffuse * textureBindless2D(mtl.texDiffuse, pc.perFrame.sampler0, vtx.uv);
  if (Kd.a < 0.5)
    discard;
  vec3 n = normalize(vtx.normal);
  vec3 L = normalize(-pc.perFrame.lightDir.xyz);
  float NdotL = clamp(dot(n, L), 0.0, 1.0);
  const vec4 f0 = vec4(0.04);
  vec4 diffuse = NdotL * pc.perFrame.ambientColor * Kd * (vec4(1.0) - f0);
  vec3 checkerTint = vec3(1.0);
  out_FragColor = bDrawNormals ?
    vec4(0.5 * (n+vec3(1.0)), 1.0) :
    Ka + diffuse * shadow(vtx.worldPos.xyz, vtx.worldPos.w, checkerTint);
  out_FragColor.rgb *= checkerTint;
};
)";

const char* kShadowVS = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 proj;
  mat4 view;
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
} pc;

void main() {
  mat4 proj  = pc.perFrame.proj;
  mat4 view  = pc.perFrame.view;
  mat4 model = pc.perObject.model;
  gl_Position = proj * view * model * vec4(pos, 1.0);
  // Pancaking: flatten casters in front of the near plane onto it instead of
  // letting them clip away, so off-frustum occluders still cast. Vulkan's clip
  // near plane is z=0 (GLM_FORCE_DEPTH_ZERO_TO_ONE); ortho w=1 so z is NDC depth.
  gl_Position.z = max(gl_Position.z, 0.0);
}
)";

const char* kShadowFS = R"(
void main() {
};
)";

const char* kContinuousShadowVS = R"(
layout (location=0) in vec3 pos;

layout(std430, buffer_reference) readonly buffer PerFrame {
  mat4 lightViewProj;
  vec4 warpParams;  // x = aNear, y = aFar, z = wNear, w = wFar
  float warpRatio;  // far / near
};

layout(std430, buffer_reference) readonly buffer PerObject {
  mat4 model;
};

layout(push_constant) uniform constants {
  PerFrame perFrame;
  PerObject perObject;
} pc;
)" WARP_LIGHT_NDC_GLSL R"(
void main() {
  vec4 worldPos = pc.perObject.model * vec4(pos, 1.0);
  vec4 lp = pc.perFrame.lightViewProj * worldPos;          // ortho -> w = 1
  vec2 warped = warpLightNDC(lp.xy / lp.w,
      pc.perFrame.warpParams.x, pc.perFrame.warpParams.y,
      pc.perFrame.warpParams.z, pc.perFrame.warpParams.w, pc.perFrame.warpRatio);
  gl_Position = vec4(warped, lp.z, 1.0);
  // Pancaking: flatten casters in front of the near plane onto it (see kShadowVS).
  gl_Position.z = max(gl_Position.z, 0.0);
}
)";

const char* kCodeFullscreenVS = R"(
layout (location=0) out vec2 uv;
void main() {
  uv = vec2((gl_VertexIndex << 1) & 2, gl_VertexIndex & 2);
  gl_Position = vec4(uv * vec2(2, -2) + vec2(-1, 1), 0.0, 1.0);
}
)";

const char* kCodeFullscreenFS = R"(
layout (location=0) in vec2 uv;
layout (location=0) out vec4 out_FragColor;

layout(push_constant) uniform constants {
  uint tex;
} pc;

void main() {
  out_FragColor = textureBindless2D(pc.tex, 0, uv);
}
)";
