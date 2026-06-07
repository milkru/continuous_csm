#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <string>

#define GLM_ENABLE_EXPERIMENTAL
#define GLM_FORCE_DEPTH_ZERO_TO_ONE
#include <glm/ext.hpp>
#include <glm/glm.hpp>

#include <taskflow/taskflow.hpp>

#include <lvk/HelpersImGui.h>
#include <lvk/LVK.h>

#include <GLFW/glfw3.h>

#include "camera.h"
#include "gui.h"
#include "model.h"
#include "shaders.h"
#include "utils.h"

constexpr bool kPreferIntegratedGPU = false;
constexpr int32_t kWindowWidth = 1920;
constexpr int32_t kWindowHeight = 1080;
constexpr glm::vec4 kAmbientColor = glm::vec4(0.82f, 0.92f, 0.98f, 1.0f);
constexpr uint32_t kNumCascades = 4;
constexpr float kCascadeSplitLambda = 0.95f;
constexpr uint32_t kShadowMapSize = 512;
// Per-cascade hardware rasterizer depth bias, applied while rendering each cascade
// into its depth map. Index 0 is the nearest (tightest) cascade; farther cascades
// cover more world space per shadow texel and need progressively more bias to stay
// free of acne without over-biasing the near cascade into peter-panning. These are
// starting points and meant to be tuned (Z_F32 may want larger values than UNORM).
constexpr float kCascadeDepthBiasConstant[kNumCascades] = { 1.25f, 1.75f, 2.5f, 3.5f };
constexpr float kCascadeDepthBiasSlope[kNumCascades] = { 1.75f, 2.0f, 2.75f, 3.5f };
const glm::vec3 kLightDir = glm::normalize(glm::vec3(0.5f, -1.0f, 0.3f));

#if defined(NDEBUG)
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

int32_t windowWidth = 0;
int32_t windowHeight = 0;

std::unique_ptr<lvk::IContext> context;
std::unique_ptr<lvk::ImGuiRenderer> imguiRenderer;
std::string folderThirdParty;
std::string folderContentRoot;

lvk::Holder<lvk::QueryPoolHandle> queryPoolTimestamps;
uint64_t pipelineTimestamps[GPUTimestamp_NUM_TIMESTAMPS] = {};
double timestampBeginRendering = 0;
double timestampEndRendering = 0;

lvk::Framebuffer framebufferMain;
lvk::Framebuffer framebufferOffscreen;
lvk::Holder<lvk::TextureHandle> framebufferOffscreenColor;
lvk::Holder<lvk::TextureHandle> framebufferOffscreenDepth;
lvk::Holder<lvk::TextureHandle> shadowCascadeTextures[kNumCascades];
lvk::Framebuffer framebufferShadowCascades[kNumCascades];
lvk::Holder<lvk::TextureHandle> shadowContinuousTexture;
lvk::Framebuffer framebufferShadowContinuous;

lvk::Holder<lvk::ShaderModuleHandle> shaderMeshVert;
lvk::Holder<lvk::ShaderModuleHandle> shaderMeshFrag;
lvk::Holder<lvk::ShaderModuleHandle> shaderMeshWireframeVert;
lvk::Holder<lvk::ShaderModuleHandle> shaderMeshWireframeFrag;
lvk::Holder<lvk::ShaderModuleHandle> shaderShadowVert;
lvk::Holder<lvk::ShaderModuleHandle> shaderShadowFrag;
lvk::Holder<lvk::ShaderModuleHandle> shaderFullscreenVert;
lvk::Holder<lvk::ShaderModuleHandle> shaderFullscreenFrag;

lvk::Holder<lvk::RenderPipelineHandle> pipelineMesh;
lvk::Holder<lvk::RenderPipelineHandle> pipelineMeshNormals;
lvk::Holder<lvk::RenderPipelineHandle> pipelineMeshWireframe;
lvk::Holder<lvk::RenderPipelineHandle> pipelineShadow;
lvk::Holder<lvk::RenderPipelineHandle> pipelineFullscreen;

lvk::Holder<lvk::BufferHandle> vertexBuffer;
lvk::Holder<lvk::BufferHandle> indexBuffer;
lvk::Holder<lvk::BufferHandle> materialsBuffer;
lvk::Holder<lvk::BufferHandle> perFrameBuffer;
lvk::Holder<lvk::BufferHandle> perFrameShadowBuffer;
lvk::Holder<lvk::BufferHandle> perObjectBuffer;
lvk::Holder<lvk::SamplerHandle> samplerLinear;
lvk::Holder<lvk::SamplerHandle> samplerShadow;
lvk::Holder<lvk::TextureHandle> textureDummyWhite;

lvk::RenderPass renderPassOffscreen;
lvk::RenderPass renderPassMain;
lvk::RenderPass renderPassShadow;
lvk::DepthState depthState;

Camera camera(glm::vec3(-100, 40, -47), glm::vec3(0, 35, 0), glm::vec3(0, 1, 0));
glm::vec2 mousePosition = glm::vec2(0.0f);
bool mousePressed = false;
bool enableWireframe = false;
bool showPerfStats = true;
bool drawNormals = false;
bool useContinuousShadow = false;
bool usePCF = true;

struct UniformsPerFrame
{
	glm::mat4 proj;
	glm::mat4 view;
	glm::mat4 cascadeLightMatrices[kNumCascades];
	glm::vec4 cascadeSplitDepths;
	uint32_t texShadow[kNumCascades] = {};
	uint32_t sampler = 0;
	uint32_t samplerShadow = 0;
	uint32_t texShadowContinuous = 0;
	uint32_t shadowMode = 0; // 0 = discrete CSM, 1 = continuous single map
	glm::vec4 ambientColor = kAmbientColor;
	glm::vec4 lightDir = {};
	glm::mat4 continuousLightMatrix = glm::mat4(1.0f);
	uint32_t pcfEnabled = 1; // 0 = single hardware-compare tap, 1 = 3x3 PCF
} perFrameUniforms;

struct UniformsPerFrameShadow
{
	glm::mat4 proj;
	glm::mat4 view;
};

struct UniformsPerObject
{
	glm::mat4 model;
	glm::mat4 normal;
};

void createPipelines();
void createShadowMap();
void createOffscreenFramebuffer();

bool init(lvk::LVKwindow* window)
{
	{
		const uint32_t pixel = 0xFFFFFFFF;
		textureDummyWhite = context->createTexture(
		    {
		        .type = lvk::TextureType_2D,
		        .format = lvk::Format_R_UN8,
		        .dimensions = { 1, 1 },
		        .usage = lvk::TextureUsageBits_Sampled,
		        .components = { lvk::Swizzle_1, lvk::Swizzle_1, lvk::Swizzle_1, lvk::Swizzle_1 },
		        .data = &pixel,
		        .debugName = "dummy 1x1 (white)",
		    },
		    nullptr);
	}

	perFrameBuffer = context->createBuffer({
	    .usage = lvk::BufferUsageBits_Uniform,
	    .storage = lvk::StorageType_HostVisible,
	    .size = sizeof(UniformsPerFrame),
	    .debugName = "Buffer: uniforms (per frame)",
	});
	perFrameShadowBuffer = context->createBuffer({
	    .usage = lvk::BufferUsageBits_Uniform,
	    .storage = lvk::StorageType_HostVisible,
	    .size = sizeof(UniformsPerFrameShadow),
	    .debugName = "Buffer: uniforms (per frame shadow)",
	});
	perObjectBuffer = context->createBuffer({
	    .usage = lvk::BufferUsageBits_Uniform,
	    .storage = lvk::StorageType_HostVisible,
	    .size = sizeof(UniformsPerObject),
	    .debugName = "Buffer: uniforms (per object)",
	});

	depthState = { .compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true };

	samplerLinear = context->createSampler({
	    .mipMap = lvk::SamplerMip_Linear,
	    .wrapU = lvk::SamplerWrap_Repeat,
	    .wrapV = lvk::SamplerWrap_Repeat,
	    .debugName = "Sampler: linear",
	});
	samplerShadow = context->createSampler({
	    .wrapU = lvk::SamplerWrap_Clamp,
	    .wrapV = lvk::SamplerWrap_Clamp,
	    .depthCompareOp = lvk::CompareOp_LessEqual,
	    .depthCompareEnabled = true,
	    .debugName = "Sampler: shadow",
	});

	renderPassOffscreen = {
	    .color = {{
	        .loadOp = lvk::LoadOp_Clear,
	        .storeOp = lvk::StoreOp_Store,
	        .clearColor = { kAmbientColor.r, kAmbientColor.g, kAmbientColor.b, 1.0f },
	    }},
	    .depth = {
	        .loadOp = lvk::LoadOp_Clear,
	        .storeOp = lvk::StoreOp_Store,
	        .clearDepth = 1.0f,
	    },
	};
	renderPassMain = {
		.color = { { .loadOp = lvk::LoadOp_Clear,
		             .storeOp = lvk::StoreOp_Store,
		             .clearColor = { 0.0f, 0.0f, 0.0f, 1.0f } } },
	};
	renderPassShadow = {
		.color = {},
		.depth = { .loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f },
	};

	framebufferMain = {
		.color = { { .texture = context->getCurrentSwapchainTexture() } },
	};

	createShadowMap();
	createOffscreenFramebuffer();
	createPipelines();

	imguiRenderer = std::make_unique<lvk::ImGuiRenderer>(
	    *context, window, (folderThirdParty + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(),
	    float(windowHeight) / 70.0f);

	queryPoolTimestamps = context->createQueryPool(GPUTimestamp_NUM_TIMESTAMPS, "queryPoolTimestamps");

	if (!initModel())
	{
		return false;
	}

	loadMaterials();

	return true;
}

void destroy()
{
	imguiRenderer = nullptr;

	vertexBuffer = nullptr;
	indexBuffer = nullptr;
	materialsBuffer = nullptr;
	perFrameBuffer = nullptr;
	perFrameShadowBuffer = nullptr;
	perObjectBuffer = nullptr;
	shaderMeshVert = nullptr;
	shaderMeshFrag = nullptr;
	shaderMeshWireframeVert = nullptr;
	shaderMeshWireframeFrag = nullptr;
	shaderShadowVert = nullptr;
	shaderShadowFrag = nullptr;
	shaderFullscreenVert = nullptr;
	shaderFullscreenFrag = nullptr;
	pipelineMesh = nullptr;
	pipelineMeshNormals = nullptr;
	pipelineMeshWireframe = nullptr;
	pipelineShadow = nullptr;
	pipelineFullscreen = nullptr;
	textureDummyWhite = nullptr;
	textures.clear();
	texturesCache.clear();
	samplerLinear = nullptr;
	samplerShadow = nullptr;
	context->destroy(framebufferMain);
	for (uint32_t i = 0; i < kNumCascades; i++)
	{
		shadowCascadeTextures[i] = nullptr;
	}
	shadowContinuousTexture = nullptr;
	framebufferOffscreenColor = nullptr;
	framebufferOffscreenDepth = nullptr;
	queryPoolTimestamps = nullptr;
	context = nullptr;

	printf("Waiting for the loader thread to exit...\n");
	loaderPool = nullptr;
}

void createPipelines()
{
	if (pipelineMesh.valid())
	{
		return;
	}

	const lvk::VertexInput vertexInputMesh = {
	    .attributes =
	        {
	            { .location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position) },
	            { .location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv) },
	            { .location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal) },
	            { .location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex) },
	        },
	    .inputBindings = { { .stride = sizeof(VertexData) } },
	};

	const lvk::VertexInput vertexInputPositionOnly = {
		.attributes = { { .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position) } },
		.inputBindings = { { .stride = sizeof(VertexData) } },
	};

	shaderMeshVert = context->createShaderModule({ kCodeVS, lvk::Stage_Vert, "Shader Module: main (vert)" });
	shaderMeshFrag = context->createShaderModule({ kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)" });
	shaderMeshWireframeVert =
	    context->createShaderModule({ kCodeVS_Wireframe, lvk::Stage_Vert, "Shader Module: main wireframe (vert)" });
	shaderMeshWireframeFrag =
	    context->createShaderModule({ kCodeFS_Wireframe, lvk::Stage_Frag, "Shader Module: main wireframe (frag)" });
	shaderShadowVert = context->createShaderModule({ kShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)" });
	shaderShadowFrag = context->createShaderModule({ kShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)" });
	shaderFullscreenVert =
	    context->createShaderModule({ kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)" });
	shaderFullscreenFrag =
	    context->createShaderModule({ kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)" });

	{
		lvk::RenderPipelineDesc desc = {
			.vertexInput = vertexInputMesh,
			.smVert = shaderMeshVert,
			.smFrag = shaderMeshFrag,
			.color = { { .format = context->getFormat(framebufferOffscreen.color[0].texture) } },
			.depthFormat = context->getFormat(framebufferOffscreen.depthStencil.texture),
			.cullMode = lvk::CullMode_Back,
			.frontFace = lvk::WindingMode_CCW,
			.debugName = "Pipeline: mesh",
		};
		pipelineMesh = context->createRenderPipeline(desc, nullptr);

		const uint32_t drawNormalsSpec = 1;
		desc.specInfo = { .entries = { { .constantId = 0, .size = sizeof(uint32_t) } },
			              .data = &drawNormalsSpec,
			              .dataSize = sizeof(drawNormalsSpec) };
		pipelineMeshNormals = context->createRenderPipeline(desc, nullptr);

		desc.specInfo = {};
		desc.polygonMode = lvk::PolygonMode_Line;
		desc.vertexInput = vertexInputPositionOnly;
		desc.smVert = shaderMeshWireframeVert;
		desc.smFrag = shaderMeshWireframeFrag;
		desc.debugName = "Pipeline: mesh (wireframe)";
		pipelineMeshWireframe = context->createRenderPipeline(desc, nullptr);
	}

	pipelineShadow = context->createRenderPipeline(
	    lvk::RenderPipelineDesc{
	        .vertexInput = vertexInputPositionOnly,
	        .smVert = shaderShadowVert,
	        .smFrag = shaderShadowFrag,
	        .depthFormat = lvk::Format_Z_F32,
	        .cullMode = lvk::CullMode_None,
	        .debugName = "Pipeline: shadow",
	    },
	    nullptr);

	{
		const lvk::RenderPipelineDesc desc = {
			.smVert = shaderFullscreenVert,
			.smFrag = shaderFullscreenFrag,
			.color = { { .format = context->getFormat(framebufferMain.color[0].texture) } },
			.depthFormat = context->getFormat(framebufferMain.depthStencil.texture),
			.cullMode = lvk::CullMode_None,
			.debugName = "Pipeline: fullscreen",
		};
		pipelineFullscreen = context->createRenderPipeline(desc, nullptr);
	}
}

void createShadowMap()
{
	const lvk::TextureDesc desc = {
		.type = lvk::TextureType_2D,
		.format = lvk::Format_Z_F32,
		.dimensions = { kShadowMapSize, kShadowMapSize },
		.usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
		.numMipLevels = 1,
	};
	for (uint32_t i = 0; i < kNumCascades; i++)
	{
		char name[32];
		snprintf(name, sizeof(name), "Shadow map (cascade %u)", i);
		shadowCascadeTextures[i] = context->createTexture(desc, name);
		framebufferShadowCascades[i] = {
			.depthStencil = { .texture = shadowCascadeTextures[i] },
		};
	}

	// Single shadow map for the continuous (non-cascaded) path.
	shadowContinuousTexture = context->createTexture(desc, "Shadow map (continuous)");
	framebufferShadowContinuous = {
		.depthStencil = { .texture = shadowContinuousTexture },
	};
}

void createOffscreenFramebuffer()
{
	const uint32_t w = (uint32_t)windowWidth;
	const uint32_t h = (uint32_t)windowHeight;

	const lvk::TextureDesc descColor = {
		.type = lvk::TextureType_2D,
		.format = lvk::Format_RGBA_UN8,
		.dimensions = { w, h },
		.usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
		.numMipLevels = 1,
		.debugName = "Offscreen framebuffer (color)",
	};
	const lvk::TextureDesc descDepth = {
		.type = lvk::TextureType_2D,
		.format = lvk::Format_Z_UN24,
		.dimensions = { w, h },
		.usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
		.numMipLevels = 1,
		.debugName = "Offscreen framebuffer (depth)",
	};

	framebufferOffscreenColor = context->createTexture(descColor);
	framebufferOffscreenDepth = context->createTexture(descDepth);
	framebufferOffscreen = {
		.color = { { .texture = framebufferOffscreenColor } },
		.depthStencil = { .texture = framebufferOffscreenDepth },
	};
}

void resize()
{
	if (!windowWidth || !windowHeight)
	{
		return;
	}
	context->recreateSwapchain(windowWidth, windowHeight);
	createOffscreenFramebuffer();
}

void render(double delta)
{
	if (!windowWidth || !windowHeight)
	{
		return;
	}

	lvk::TextureHandle nativeDrawable = context->getCurrentSwapchainTexture();
	framebufferMain.color[0].texture = nativeDrawable;

	{
		imguiRenderer->beginFrame(framebufferMain);

		ImGui::Begin("Keyboard Hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
		ImGui::Text("W/S/A/D - Camera Movement");
		ImGui::Text("Q/E - Camera Up/Down");
		ImGui::Text("Shift - Fast Movement");
		ImGui::Text("N - Toggle Normals");
		ImGui::Text("T - Toggle Wireframe");
		ImGui::Text("P - Show Performance Stats");
		ImGui::Text("C - Continuous Shadow (%s)", useContinuousShadow ? "ON" : "OFF");
		ImGui::Text("F - PCF Filtering (%s)", usePCF ? "ON" : "OFF");
		ImGui::End();

		if (uint32_t materialsRemaining = remainingMaterialsToLoad.load(std::memory_order_acquire))
		{
			ImGui::SetNextWindowPos(ImVec2(0, 0));
			ImGui::Begin("Loading...", nullptr,
			             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNavInputs);
			ImGui::ProgressBar(1.0f - float(materialsRemaining) / cachedMaterials.size(),
			                   ImVec2(ImGui::GetIO().DisplaySize.x, 32));
			ImGui::End();
		}

		if (showPerfStats)
		{
			showTimeGPU();
		}
	}

	camera.update(delta, mousePosition, mousePressed);

	timestampBeginRendering = glfwGetTime();

	const float fov = float(45.0f * (M_PI / 180.0f));
	const float aspectRatio = (float)windowWidth / (float)windowHeight;
	const glm::mat4 proj = glm::perspective(fov, aspectRatio, 0.5f, 500.0f);
	const glm::mat4 view = camera.getViewMatrix();

	// Compute cascade split depths and light matrices
	const float nearClip = 0.5f;
	const float farClip = 500.0f;
	const float clipRange = farClip - nearClip;
	const glm::vec3 lightDir = kLightDir;
	const glm::mat4 inverseViewProj = glm::inverse(proj * view);
	const glm::mat4 scaleBias(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);
	const glm::mat4 modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.05f));

	float cascadeSplits[kNumCascades];
	{
		const float ratio = farClip / nearClip;
		for (uint32_t i = 0; i < kNumCascades; i++)
		{
			float splitFraction = (i + 1) / float(kNumCascades);
			float logSplit = nearClip * std::pow(ratio, splitFraction);
			float uniformSplit = nearClip + clipRange * splitFraction;
			float splitDepth = kCascadeSplitLambda * (logSplit - uniformSplit) + uniformSplit;
			cascadeSplits[i] = (splitDepth - nearClip) / clipRange;
		}
	}

	glm::mat4 shadowProjs[kNumCascades];
	glm::mat4 shadowViews[kNumCascades];

	float lastSplitDist = 0.0f;
	for (uint32_t i = 0; i < kNumCascades; i++)
	{
		float splitDist = cascadeSplits[i];

		glm::vec3 frustumCorners[8] = {
			{ -1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, -1.0f, 0.0f }, { -1.0f, -1.0f, 0.0f },
			{ -1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, -1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
		};
		for (uint32_t j = 0; j < 8; j++)
		{
			glm::vec4 corner = inverseViewProj * glm::vec4(frustumCorners[j], 1.0f);
			frustumCorners[j] = glm::vec3(corner) / corner.w;
		}
		for (uint32_t j = 0; j < 4; j++)
		{
			glm::vec3 dist = frustumCorners[j + 4] - frustumCorners[j];
			frustumCorners[j + 4] = frustumCorners[j] + dist * splitDist;
			frustumCorners[j] = frustumCorners[j] + dist * lastSplitDist;
		}

		glm::vec3 center(0.0f);
		for (uint32_t j = 0; j < 8; j++)
		{
			center += frustumCorners[j];
		}
		center /= 8.0f;

		float radius = 0.0f;
		for (uint32_t j = 0; j < 8; j++)
		{
			radius = glm::max(radius, glm::length(frustumCorners[j] - center));
		}
		radius = std::ceil(radius * 16.0f) / 16.0f;

		// Tight sphere-fit depth range keeps depth precision high. Casters in front
		// of the near plane (between the light and the slice) are not lost: they are
		// flattened onto the near plane via "pancaking" in the shadow vertex shader,
		// since LVK exposes no hardware depth-clamp toggle.
		const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
		shadowViews[i] = glm::lookAt(center - lightDir * radius, center, up);
		shadowProjs[i] = glm::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * radius);

		// Stabilize: snap the projection to whole shadow-map texels so shadow edges
		// don't shimmer as the camera moves (Valient stable CSM / MJP). Radius is
		// rotation-invariant for a symmetric frustum slice, so the projection size
		// is constant per cascade and only the origin needs snapping.
		const glm::mat4 shadowMatrix = shadowProjs[i] * shadowViews[i];
		glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		shadowOrigin *= float(kShadowMapSize) * 0.5f;
		const glm::vec2 roundedOrigin = glm::round(glm::vec2(shadowOrigin));
		glm::vec2 roundOffset = roundedOrigin - glm::vec2(shadowOrigin);
		roundOffset *= 2.0f / float(kShadowMapSize);
		shadowProjs[i][3][0] += roundOffset.x;
		shadowProjs[i][3][1] += roundOffset.y;

		perFrameUniforms.cascadeLightMatrices[i] = scaleBias * shadowProjs[i] * shadowViews[i];
		perFrameUniforms.cascadeSplitDepths[i] = -(nearClip + splitDist * clipRange);
		perFrameUniforms.texShadow[i] = shadowCascadeTextures[i].index();

		lastSplitDist = splitDist;
	}

	// Single light-space shadow map fit to the whole camera frustum (continuous path,
	// checkpoint 1a: no warp yet). Same sphere-fit + texel-snap stabilization as a
	// cascade, but covering the entire near->far range as one map.
	glm::mat4 continuousLightProj;
	glm::mat4 continuousLightView;
	{
		glm::vec3 frustumCorners[8] = {
			{ -1.0f, 1.0f, 0.0f }, { 1.0f, 1.0f, 0.0f }, { 1.0f, -1.0f, 0.0f }, { -1.0f, -1.0f, 0.0f },
			{ -1.0f, 1.0f, 1.0f }, { 1.0f, 1.0f, 1.0f }, { 1.0f, -1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
		};
		for (uint32_t j = 0; j < 8; j++)
		{
			glm::vec4 corner = inverseViewProj * glm::vec4(frustumCorners[j], 1.0f);
			frustumCorners[j] = glm::vec3(corner) / corner.w;
		}

		glm::vec3 center(0.0f);
		for (uint32_t j = 0; j < 8; j++)
		{
			center += frustumCorners[j];
		}
		center /= 8.0f;

		float radius = 0.0f;
		for (uint32_t j = 0; j < 8; j++)
		{
			radius = glm::max(radius, glm::length(frustumCorners[j] - center));
		}
		radius = std::ceil(radius * 16.0f) / 16.0f;

		const glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f);
		continuousLightView = glm::lookAt(center - lightDir * radius, center, up);
		continuousLightProj = glm::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * radius);

		const glm::mat4 shadowMatrix = continuousLightProj * continuousLightView;
		glm::vec4 shadowOrigin = shadowMatrix * glm::vec4(0.0f, 0.0f, 0.0f, 1.0f);
		shadowOrigin *= float(kShadowMapSize) * 0.5f;
		const glm::vec2 roundedOrigin = glm::round(glm::vec2(shadowOrigin));
		glm::vec2 roundOffset = roundedOrigin - glm::vec2(shadowOrigin);
		roundOffset *= 2.0f / float(kShadowMapSize);
		continuousLightProj[3][0] += roundOffset.x;
		continuousLightProj[3][1] += roundOffset.y;

		perFrameUniforms.continuousLightMatrix = scaleBias * continuousLightProj * continuousLightView;
	}

	perFrameUniforms.texShadowContinuous = shadowContinuousTexture.index();
	perFrameUniforms.shadowMode = useContinuousShadow ? 1u : 0u;
	perFrameUniforms.pcfEnabled = usePCF ? 1u : 0u;

	perFrameUniforms.proj = proj;
	perFrameUniforms.view = view;
	perFrameUniforms.sampler = samplerLinear.index();
	perFrameUniforms.samplerShadow = samplerShadow.index();
	perFrameUniforms.ambientColor = kAmbientColor;
	perFrameUniforms.lightDir = glm::vec4(kLightDir, 0.0f);

	const UniformsPerObject perObject = {
		.model = modelMatrix,
		.normal = glm::transpose(glm::inverse(modelMatrix)),
	};

	lvk::ICommandBuffer& buffer = context->acquireCommandBuffer();

	processLoadedMaterials(buffer);

	buffer.cmdUpdateBuffer(perFrameBuffer, 0, sizeof(perFrameUniforms), &perFrameUniforms);
	buffer.cmdUpdateBuffer(perObjectBuffer, 0, sizeof(perObject), &perObject);

	// Pass 1: shadows (CSM: one sub-pass per cascade; continuous: one single map)
	{
		struct
		{
			uint64_t perFrame;
			uint64_t perObject;
		} bindings = {
			.perFrame = context->gpuAddress(perFrameShadowBuffer),
			.perObject = context->gpuAddress(perObjectBuffer),
		};

		if (useContinuousShadow)
		{
			const UniformsPerFrameShadow perFrameShadow{
				.proj = continuousLightProj,
				.view = continuousLightView,
			};
			buffer.cmdUpdateBuffer(perFrameShadowBuffer, 0, sizeof(perFrameShadow), &perFrameShadow);
			buffer.cmdBeginRendering(renderPassShadow, framebufferShadowContinuous);
			{
				buffer.cmdBindRenderPipeline(pipelineShadow);
				buffer.cmdPushDebugGroupLabel("Render Shadows (continuous)", 0xff0000ff);
				buffer.cmdBindDepthState(depthState);
				buffer.cmdSetDepthBiasEnable(true);
				// One coarse map over the whole frustum -> far-cascade-sized texels, so
				// start from the largest cascade bias (tunable).
				buffer.cmdSetDepthBias(kCascadeDepthBiasConstant[kNumCascades - 1], kCascadeDepthBiasSlope[kNumCascades - 1]);
				buffer.cmdBindVertexBuffer(0, vertexBuffer, 0);
				buffer.cmdPushConstants(bindings);
				buffer.cmdBindIndexBuffer(indexBuffer, lvk::IndexFormat_UI32);
				buffer.cmdDrawIndexed((uint32_t)indexData.size());
				buffer.cmdPopDebugGroupLabel();
			}
			buffer.cmdEndRendering();
			buffer.cmdTransitionToShaderReadOnly({ shadowContinuousTexture }, {});
		}
		else
		for (uint32_t i = 0; i < kNumCascades; i++)
		{
			const UniformsPerFrameShadow perFrameShadow{
				.proj = shadowProjs[i],
				.view = shadowViews[i],
			};
			buffer.cmdUpdateBuffer(perFrameShadowBuffer, 0, sizeof(perFrameShadow), &perFrameShadow);
			buffer.cmdBeginRendering(renderPassShadow, framebufferShadowCascades[i]);
			{
				buffer.cmdBindRenderPipeline(pipelineShadow);
				buffer.cmdPushDebugGroupLabel("Render Shadows", 0xff0000ff);
				buffer.cmdBindDepthState(depthState);
				// Per-cascade slope-scaled depth bias, baked into the shadow map.
				// (LVK resets depth-bias-enable on every cmdBeginRendering, so this
				// must be set inside each cascade's pass.)
				buffer.cmdSetDepthBiasEnable(true);
				buffer.cmdSetDepthBias(kCascadeDepthBiasConstant[i], kCascadeDepthBiasSlope[i]);
				buffer.cmdBindVertexBuffer(0, vertexBuffer, 0);
				buffer.cmdPushConstants(bindings);
				buffer.cmdBindIndexBuffer(indexBuffer, lvk::IndexFormat_UI32);
				buffer.cmdDrawIndexed((uint32_t)indexData.size());
				buffer.cmdPopDebugGroupLabel();
			}
			buffer.cmdEndRendering();
			buffer.cmdTransitionToShaderReadOnly({ shadowCascadeTextures[i] }, {});
		}
	}

#define GPU_TIMESTAMP(ts) buffer.cmdWriteTimestamp(queryPoolTimestamps, ts);

	// Pass 2: mesh
	{
		buffer.cmdResetQueryPool(queryPoolTimestamps, 0, GPUTimestamp_NUM_TIMESTAMPS);
		GPU_TIMESTAMP(GPUTimestamp_BeginSceneRendering);

		buffer.cmdBeginRendering(renderPassOffscreen, framebufferOffscreen);
		{
			buffer.cmdBindRenderPipeline(drawNormals ? pipelineMeshNormals : pipelineMesh);
			buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
			buffer.cmdBindDepthState(depthState);
			buffer.cmdBindVertexBuffer(0, vertexBuffer, 0);

			struct
			{
				uint64_t perFrame;
				uint64_t perObject;
				uint64_t materials;
			} bindings = {
				.perFrame = context->gpuAddress(perFrameBuffer),
				.perObject = context->gpuAddress(perObjectBuffer),
				.materials = context->gpuAddress(materialsBuffer),
			};

			buffer.cmdPushConstants(bindings);
			buffer.cmdBindIndexBuffer(indexBuffer, lvk::IndexFormat_UI32);
			buffer.cmdDrawIndexed((uint32_t)indexData.size());
			if (enableWireframe)
			{
				buffer.cmdBindRenderPipeline(pipelineMeshWireframe);
				buffer.cmdDrawIndexed((uint32_t)indexData.size());
			}
			buffer.cmdPopDebugGroupLabel();
		}
		buffer.cmdEndRendering();

		GPU_TIMESTAMP(GPUTimestamp_EndSceneRendering);
	}

	// Pass 3: fullscreen blit to swapchain
	{
		GPU_TIMESTAMP(GPUTimestamp_BeginPresent);

		const lvk::TextureHandle offscreenTexture = framebufferOffscreen.color[0].texture;
		buffer.cmdBeginRendering(renderPassMain, framebufferMain, { .sampledImages = { offscreenTexture } });
		{
			buffer.cmdBindRenderPipeline(pipelineFullscreen);
			buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
			buffer.cmdBindDepthState(depthState);

			struct
			{
				uint32_t texture;
			} bindings = {
				.texture = offscreenTexture.index(),
			};

			buffer.cmdPushConstants(bindings);
			buffer.cmdDraw(3);
			buffer.cmdPopDebugGroupLabel();

			imguiRenderer->endFrame(buffer);
		}
		buffer.cmdEndRendering();

		GPU_TIMESTAMP(GPUTimestamp_EndPresent);
	}

#undef GPU_TIMESTAMP

	context->submit(buffer, framebufferMain.color[0].texture);

	timestampEndRendering = glfwGetTime();

	if (showPerfStats)
	{
		context->getQueryPoolResults(queryPoolTimestamps, 0, LVK_ARRAY_NUM_ELEMENTS(pipelineTimestamps),
		                             sizeof(pipelineTimestamps), pipelineTimestamps, sizeof(pipelineTimestamps[0]));
	}
}

GLFWkeyfun g_PrevKeyCallback = nullptr;
GLFWmousebuttonfun g_PrevMouseButtonCallback = nullptr;

int main(int argc, char* argv[])
{
#if defined(LVK_WITH_MINILOG)
	minilog::initialize(nullptr, { .threadNames = false });
#endif

	{
		using namespace std::filesystem;
		const path subdir("content/");
		path dir = current_path();
		while (dir != current_path().root_path() && !exists(dir / subdir))
		{
			dir = dir.parent_path();
		}
		if (!exists(dir / subdir))
		{
			printf("Cannot find the content directory. Run `download_content.py` before running this app.");
			LVK_ASSERT(false);
			return EXIT_FAILURE;
		}
		folderThirdParty = (dir / path("3rdparty/lightweightvk/third-party/deps/src/")).string();
		folderContentRoot = (dir / subdir).string();
	}

	windowWidth = kWindowWidth;
	windowHeight = kWindowHeight;
	lvk::LVKwindow* window = lvk::initWindow("Continuous CSM", windowWidth, windowHeight, true);

	// Center the window on the primary monitor's work area (taskbar excluded)
	if (window)
	{
		if (GLFWmonitor* monitor = glfwGetPrimaryMonitor())
		{
			int32_t areaX = 0, areaY = 0, areaW = 0, areaH = 0;
			glfwGetMonitorWorkarea(monitor, &areaX, &areaY, &areaW, &areaH);
			glfwSetWindowPos(window, areaX + (areaW - windowWidth) / 2, areaY + (areaH - windowHeight) / 2);
		}
	}

	context = lvk::createVulkanContextWithSwapchain(window, windowWidth, windowHeight,
	                                                {
	                                                    .enableValidation = kEnableValidationLayers,
	                                                },
	                                                kPreferIntegratedGPU ? lvk::HWDeviceType_Integrated
	                                                                     : lvk::HWDeviceType_Discrete);
	if (!context)
	{
		return EXIT_FAILURE;
	}

	if (!init(window))
	{
		return EXIT_FAILURE;
	}

	double prevTime = glfwGetTime();

	glfwSetFramebufferSizeCallback(window,
	                               [](GLFWwindow*, int32_t newWidth, int32_t newHeight)
	                               {
		                               windowWidth = newWidth;
		                               windowHeight = newHeight;
		                               resize();
	                               });

	glfwSetCursorPosCallback(window,
	                         [](GLFWwindow* w, double x, double y)
	                         {
		                         int32_t width, height;
		                         glfwGetFramebufferSize(w, &width, &height);
		                         if (width && height)
		                         {
			                         mousePosition = glm::vec2(x / width, 1.0f - y / height);
			                         ImGui::GetIO().MousePos = ImVec2((float)x, (float)y);
		                         }
	                         });

	g_PrevMouseButtonCallback = glfwSetMouseButtonCallback(
	    window,
	    [](GLFWwindow* w, int32_t button, int32_t action, int32_t mods)
	    {
		    if (!ImGui::GetIO().WantCaptureMouse)
		    {
			    if (button == GLFW_MOUSE_BUTTON_RIGHT)
			    {
				    mousePressed = (action == GLFW_PRESS);
				    glfwSetInputMode(w, GLFW_CURSOR, mousePressed ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
			    }
		    }
		    else
		    {
			    mousePressed = false;
		    }
		    if (g_PrevMouseButtonCallback)
		    {
			    g_PrevMouseButtonCallback(w, button, action, mods);
		    }
	    });

	g_PrevKeyCallback =
	    glfwSetKeyCallback(window,
	                       [](GLFWwindow* w, int32_t key, int32_t scancode, int32_t action, int32_t mods)
	                       {
		                       const bool pressed = action != GLFW_RELEASE && !ImGui::GetIO().WantCaptureKeyboard;
		                       if (key == GLFW_KEY_ESCAPE && pressed)
		                       {
			                       loaderShouldExit.store(true, std::memory_order_release);
			                       glfwSetWindowShouldClose(w, GLFW_TRUE);
		                       }
		                       if (key == GLFW_KEY_N && pressed)
		                       {
			                       drawNormals = !drawNormals;
		                       }
		                       if (key == GLFW_KEY_T && pressed)
		                       {
			                       enableWireframe = !enableWireframe;
		                       }
		                       if (key == GLFW_KEY_P && pressed)
		                       {
			                       showPerfStats = !showPerfStats;
		                       }
		                       if (key == GLFW_KEY_C && pressed)
		                       {
			                       useContinuousShadow = !useContinuousShadow;
		                       }
		                       if (key == GLFW_KEY_F && pressed)
		                       {
			                       usePCF = !usePCF;
		                       }
		                       if (key == GLFW_KEY_W)
		                       {
			                       camera.movement.forward = pressed;
		                       }
		                       if (key == GLFW_KEY_S)
		                       {
			                       camera.movement.backward = pressed;
		                       }
		                       if (key == GLFW_KEY_A)
		                       {
			                       camera.movement.left = pressed;
		                       }
		                       if (key == GLFW_KEY_D)
		                       {
			                       camera.movement.right = pressed;
		                       }
		                       if (key == GLFW_KEY_Q)
		                       {
			                       camera.movement.up = pressed;
		                       }
		                       if (key == GLFW_KEY_E)
		                       {
			                       camera.movement.down = pressed;
		                       }
		                       if (mods & GLFW_MOD_SHIFT)
		                       {
			                       camera.movement.fastSpeed = pressed;
		                       }
		                       if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
		                       {
			                       camera.movement.fastSpeed = pressed;
		                       }
		                       if (key == GLFW_KEY_SPACE)
		                       {
			                       camera.setUpVector(glm::vec3(0.0f, 1.0f, 0.0f));
		                       }
		                       if (g_PrevKeyCallback)
		                       {
			                       g_PrevKeyCallback(w, key, scancode, action, mods);
		                       }
	                       });

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		const double newTime = glfwGetTime();
		const double delta = newTime - prevTime;
		prevTime = newTime;

		if (!windowWidth || !windowHeight)
		{
			continue;
		}

		render(delta);
	}

	destroy();

	glfwDestroyWindow(window);
	glfwTerminate();

	return EXIT_SUCCESS;
}
