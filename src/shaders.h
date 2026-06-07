#pragma once

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
  mat4 continuousLightMatrix;
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
  mat4 continuousLightMatrix;
  uint pcfEnabled;
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

float shadow(vec3 worldPos, float viewZ) {
  uint texId;
  vec4 sc;
  if (pc.perFrame.shadowMode == 1u) {
    // Continuous: a single light-space shadow map fit to the whole frustum.
    sc = pc.perFrame.continuousLightMatrix * vec4(worldPos, 1.0);
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
  out_FragColor = bDrawNormals ?
    vec4(0.5 * (n+vec3(1.0)), 1.0) :
    Ka + diffuse * shadow(vtx.worldPos.xyz, vtx.worldPos.w);
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
