#if !defined(_USE_MATH_DEFINES)
#define _USE_MATH_DEFINES
#endif

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <filesystem>
#include <memory>
#include <stdio.h>
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
constexpr glm::vec4 kAmbientColor = glm::vec4(0.82f, 0.92f, 0.98f, 1.0f);
constexpr uint32_t kNumCascades = 4;
constexpr float kCascadeSplitLambda = 0.95f;
const glm::vec3 kLightDir = glm::normalize(glm::vec3(0.5f, -1.0f, 0.3f));

#if defined(NDEBUG)
constexpr bool kEnableValidationLayers = false;
#else
constexpr bool kEnableValidationLayers = true;
#endif

int32_t width_ = 0;
int32_t height_ = 0;

std::unique_ptr<lvk::IContext> ctx_;
std::unique_ptr<lvk::ImGuiRenderer> imgui_;
std::string folderThirdParty;
std::string folderContentRoot;

lvk::Holder<lvk::QueryPoolHandle> queryPoolTimestamps_;
uint64_t pipelineTimestamps[GPUTimestamp_NUM_TIMESTAMPS] = {};
double timestampBeginRendering = 0;
double timestampEndRendering = 0;

lvk::Framebuffer fbMain_;
lvk::Framebuffer fbOffscreen_;
lvk::Holder<lvk::TextureHandle> fbOffscreenColor_;
lvk::Holder<lvk::TextureHandle> fbOffscreenDepth_;
lvk::Holder<lvk::TextureHandle> shadowCascadeTextures_[kNumCascades];
lvk::Framebuffer fbShadowCascades_[kNumCascades];

lvk::Holder<lvk::ShaderModuleHandle> smMeshVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshWireframeVert_;
lvk::Holder<lvk::ShaderModuleHandle> smMeshWireframeFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowVert_;
lvk::Holder<lvk::ShaderModuleHandle> smShadowFrag_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenVert_;
lvk::Holder<lvk::ShaderModuleHandle> smFullscreenFrag_;

lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Mesh_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshNormals_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_MeshWireframe_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Shadow_;
lvk::Holder<lvk::RenderPipelineHandle> renderPipelineState_Fullscreen_;

lvk::Holder<lvk::BufferHandle> vb0_;
lvk::Holder<lvk::BufferHandle> ib0_;
lvk::Holder<lvk::BufferHandle> sbMaterials_;
lvk::Holder<lvk::BufferHandle> ubPerFrame_;
lvk::Holder<lvk::BufferHandle> ubPerFrameShadow_;
lvk::Holder<lvk::BufferHandle> ubPerObject_;
lvk::Holder<lvk::SamplerHandle> sampler_;
lvk::Holder<lvk::SamplerHandle> samplerShadow_;
lvk::Holder<lvk::TextureHandle> textureDummyWhite_;

lvk::RenderPass renderPassOffscreen_;
lvk::RenderPass renderPassMain_;
lvk::RenderPass renderPassShadow_;
lvk::DepthState depthState_;
lvk::DepthState depthStateLEqual_;

Camera camera_(glm::vec3(-100, 40, -47), glm::vec3(0, 35, 0), glm::vec3(0, 1, 0));
glm::vec2 mousePos_ = glm::vec2(0.0f);
bool mousePressed_ = false;
bool enableWireframe_ = false;
bool showPerfStats_ = true;
bool drawNormals_ = false;

struct UniformsPerFrame
{
	glm::mat4 proj;
	glm::mat4 view;
	glm::mat4 cascadeLightMatrices[kNumCascades];
	glm::vec4 cascadeSplitDepths;
	uint32_t texShadow[kNumCascades] = {};
	uint32_t sampler = 0;
	uint32_t samplerShadow = 0;
	uint32_t padding0 = 0;
	uint32_t padding1 = 0;
	glm::vec4 ambientColor = kAmbientColor;
} perFrame_;

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
		textureDummyWhite_ = ctx_->createTexture(
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

	ubPerFrame_ = ctx_->createBuffer({
	    .usage = lvk::BufferUsageBits_Uniform,
	    .storage = lvk::StorageType_HostVisible,
	    .size = sizeof(UniformsPerFrame),
	    .debugName = "Buffer: uniforms (per frame)",
	});
	ubPerFrameShadow_ = ctx_->createBuffer({
	    .usage = lvk::BufferUsageBits_Uniform,
	    .storage = lvk::StorageType_HostVisible,
	    .size = sizeof(UniformsPerFrameShadow),
	    .debugName = "Buffer: uniforms (per frame shadow)",
	});
	ubPerObject_ = ctx_->createBuffer({
	    .usage = lvk::BufferUsageBits_Uniform,
	    .storage = lvk::StorageType_HostVisible,
	    .size = sizeof(UniformsPerObject),
	    .debugName = "Buffer: uniforms (per object)",
	});

	depthState_ = { .compareOp = lvk::CompareOp_Less, .isDepthWriteEnabled = true };
	depthStateLEqual_ = { .compareOp = lvk::CompareOp_LessEqual, .isDepthWriteEnabled = true };

	sampler_ = ctx_->createSampler({
	    .mipMap = lvk::SamplerMip_Linear,
	    .wrapU = lvk::SamplerWrap_Repeat,
	    .wrapV = lvk::SamplerWrap_Repeat,
	    .debugName = "Sampler: linear",
	});
	samplerShadow_ = ctx_->createSampler({
	    .wrapU = lvk::SamplerWrap_Clamp,
	    .wrapV = lvk::SamplerWrap_Clamp,
	    .depthCompareOp = lvk::CompareOp_LessEqual,
	    .depthCompareEnabled = true,
	    .debugName = "Sampler: shadow",
	});

	renderPassOffscreen_ = {
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
	renderPassMain_ = {
		.color = { { .loadOp = lvk::LoadOp_Clear,
		             .storeOp = lvk::StoreOp_Store,
		             .clearColor = { 0.0f, 0.0f, 0.0f, 1.0f } } },
	};
	renderPassShadow_ = {
		.color = {},
		.depth = { .loadOp = lvk::LoadOp_Clear, .storeOp = lvk::StoreOp_Store, .clearDepth = 1.0f },
	};

	fbMain_ = {
		.color = { { .texture = ctx_->getCurrentSwapchainTexture() } },
	};

	createShadowMap();
	createOffscreenFramebuffer();
	createPipelines();

	imgui_ = std::make_unique<lvk::ImGuiRenderer>(
	    *ctx_, window, (folderThirdParty + "3D-Graphics-Rendering-Cookbook/data/OpenSans-Light.ttf").c_str(),
	    float(height_) / 70.0f);

	queryPoolTimestamps_ = ctx_->createQueryPool(GPUTimestamp_NUM_TIMESTAMPS, "queryPoolTimestamps_");

	if (!initModel())
	{
		return false;
	}

	loadMaterials();

	return true;
}

void destroy()
{
	imgui_ = nullptr;

	vb0_ = nullptr;
	ib0_ = nullptr;
	sbMaterials_ = nullptr;
	ubPerFrame_ = nullptr;
	ubPerFrameShadow_ = nullptr;
	ubPerObject_ = nullptr;
	smMeshVert_ = nullptr;
	smMeshFrag_ = nullptr;
	smMeshWireframeVert_ = nullptr;
	smMeshWireframeFrag_ = nullptr;
	smShadowVert_ = nullptr;
	smShadowFrag_ = nullptr;
	smFullscreenVert_ = nullptr;
	smFullscreenFrag_ = nullptr;
	renderPipelineState_Mesh_ = nullptr;
	renderPipelineState_MeshNormals_ = nullptr;
	renderPipelineState_MeshWireframe_ = nullptr;
	renderPipelineState_Shadow_ = nullptr;
	renderPipelineState_Fullscreen_ = nullptr;
	textureDummyWhite_ = nullptr;
	textures_.clear();
	texturesCache_.clear();
	sampler_ = nullptr;
	samplerShadow_ = nullptr;
	ctx_->destroy(fbMain_);
	for (uint32_t i = 0; i < kNumCascades; i++)
		shadowCascadeTextures_[i] = nullptr;
	fbOffscreenColor_ = nullptr;
	fbOffscreenDepth_ = nullptr;
	queryPoolTimestamps_ = nullptr;
	ctx_ = nullptr;

	printf("Waiting for the loader thread to exit...\n");
	loaderPool_ = nullptr;
}

void createPipelines()
{
	if (renderPipelineState_Mesh_.valid())
	{
		return;
	}

	const lvk::VertexInput vdesc = {
	    .attributes =
	        {
	            { .location = 0, .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position) },
	            { .location = 1, .format = lvk::VertexFormat_HalfFloat2, .offset = offsetof(VertexData, uv) },
	            { .location = 2, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, normal) },
	            { .location = 3, .format = lvk::VertexFormat_UShort1, .offset = offsetof(VertexData, mtlIndex) },
	        },
	    .inputBindings = { { .stride = sizeof(VertexData) } },
	};

	const lvk::VertexInput vdescs = {
		.attributes = { { .format = lvk::VertexFormat_Float3, .offset = offsetof(VertexData, position) } },
		.inputBindings = { { .stride = sizeof(VertexData) } },
	};

	smMeshVert_ = ctx_->createShaderModule({ kCodeVS, lvk::Stage_Vert, "Shader Module: main (vert)" });
	smMeshFrag_ = ctx_->createShaderModule({ kCodeFS, lvk::Stage_Frag, "Shader Module: main (frag)" });
	smMeshWireframeVert_ =
	    ctx_->createShaderModule({ kCodeVS_Wireframe, lvk::Stage_Vert, "Shader Module: main wireframe (vert)" });
	smMeshWireframeFrag_ =
	    ctx_->createShaderModule({ kCodeFS_Wireframe, lvk::Stage_Frag, "Shader Module: main wireframe (frag)" });
	smShadowVert_ = ctx_->createShaderModule({ kShadowVS, lvk::Stage_Vert, "Shader Module: shadow (vert)" });
	smShadowFrag_ = ctx_->createShaderModule({ kShadowFS, lvk::Stage_Frag, "Shader Module: shadow (frag)" });
	smFullscreenVert_ =
	    ctx_->createShaderModule({ kCodeFullscreenVS, lvk::Stage_Vert, "Shader Module: fullscreen (vert)" });
	smFullscreenFrag_ =
	    ctx_->createShaderModule({ kCodeFullscreenFS, lvk::Stage_Frag, "Shader Module: fullscreen (frag)" });

	{
		lvk::RenderPipelineDesc desc = {
			.vertexInput = vdesc,
			.smVert = smMeshVert_,
			.smFrag = smMeshFrag_,
			.color = { { .format = ctx_->getFormat(fbOffscreen_.color[0].texture) } },
			.depthFormat = ctx_->getFormat(fbOffscreen_.depthStencil.texture),
			.cullMode = lvk::CullMode_Back,
			.frontFace = lvk::WindingMode_CCW,
			.debugName = "Pipeline: mesh",
		};
		renderPipelineState_Mesh_ = ctx_->createRenderPipeline(desc, nullptr);

		const uint32_t drawNormals = 1;
		desc.specInfo = { .entries = { { .constantId = 0, .size = sizeof(uint32_t) } },
			              .data = &drawNormals,
			              .dataSize = sizeof(drawNormals) };
		renderPipelineState_MeshNormals_ = ctx_->createRenderPipeline(desc, nullptr);

		desc.specInfo = {};
		desc.polygonMode = lvk::PolygonMode_Line;
		desc.vertexInput = vdescs;
		desc.smVert = smMeshWireframeVert_;
		desc.smFrag = smMeshWireframeFrag_;
		desc.debugName = "Pipeline: mesh (wireframe)";
		renderPipelineState_MeshWireframe_ = ctx_->createRenderPipeline(desc, nullptr);
	}

	renderPipelineState_Shadow_ = ctx_->createRenderPipeline(
	    lvk::RenderPipelineDesc{
	        .vertexInput = vdescs,
	        .smVert = smShadowVert_,
	        .smFrag = smShadowFrag_,
	        .depthFormat = lvk::Format_Z_UN16,
	        .cullMode = lvk::CullMode_None,
	        .debugName = "Pipeline: shadow",
	    },
	    nullptr);

	{
		const lvk::RenderPipelineDesc desc = {
			.smVert = smFullscreenVert_,
			.smFrag = smFullscreenFrag_,
			.color = { { .format = ctx_->getFormat(fbMain_.color[0].texture) } },
			.depthFormat = ctx_->getFormat(fbMain_.depthStencil.texture),
			.cullMode = lvk::CullMode_None,
			.debugName = "Pipeline: fullscreen",
		};
		renderPipelineState_Fullscreen_ = ctx_->createRenderPipeline(desc, nullptr);
	}
}

void createShadowMap()
{
	const uint32_t kShadowMapSize = 4096;
	const lvk::TextureDesc desc = {
		.type = lvk::TextureType_2D,
		.format = lvk::Format_Z_UN16,
		.dimensions = { kShadowMapSize, kShadowMapSize },
		.usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
		.numMipLevels = 1,
	};
	for (uint32_t i = 0; i < kNumCascades; i++)
	{
		char name[32];
		snprintf(name, sizeof(name), "Shadow map (cascade %u)", i);
		shadowCascadeTextures_[i] = ctx_->createTexture(desc, name);
		fbShadowCascades_[i] = {
			.depthStencil = { .texture = shadowCascadeTextures_[i] },
		};
	}
}

void createOffscreenFramebuffer()
{
	const uint32_t w = (uint32_t)width_;
	const uint32_t h = (uint32_t)height_;

	const lvk::TextureDesc descColor = {
		.type = lvk::TextureType_2D,
		.format = lvk::Format_RGBA_UN8,
		.dimensions = { w, h },
		.usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled | lvk::TextureUsageBits_Storage,
		.numMipLevels = lvk::calcNumMipLevels(w, h),
		.debugName = "Offscreen framebuffer (color)",
	};
	const lvk::TextureDesc descDepth = {
		.type = lvk::TextureType_2D,
		.format = lvk::Format_Z_UN24,
		.dimensions = { w, h },
		.usage = lvk::TextureUsageBits_Attachment | lvk::TextureUsageBits_Sampled,
		.numMipLevels = lvk::calcNumMipLevels(w, h),
		.debugName = "Offscreen framebuffer (depth)",
	};

	fbOffscreenColor_ = ctx_->createTexture(descColor);
	fbOffscreenDepth_ = ctx_->createTexture(descDepth);
	fbOffscreen_ = {
		.color = { { .texture = fbOffscreenColor_ } },
		.depthStencil = { .texture = fbOffscreenDepth_ },
	};
}

void resize()
{
	if (!width_ || !height_)
	{
		return;
	}
	ctx_->recreateSwapchain(width_, height_);
	createOffscreenFramebuffer();
}

void render(double delta)
{
	if (!width_ || !height_)
	{
		return;
	}

	lvk::TextureHandle nativeDrawable = ctx_->getCurrentSwapchainTexture();
	fbMain_.color[0].texture = nativeDrawable;

	{
		imgui_->beginFrame(fbMain_);

		ImGui::Begin("Keyboard Hints:", nullptr, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoNavInputs);
		ImGui::Text("W/S/A/D - Camera Movement");
		ImGui::Text("Q/E - Camera Up/Down");
		ImGui::Text("Shift - Fast Movement");
		ImGui::Text("N - Toggle Normals");
		ImGui::Text("T - Toggle Wireframe");
		ImGui::Text("P - Show Performance Stats");
		ImGui::End();

		if (uint32_t num = remainingMaterialsToLoad_.load(std::memory_order_acquire))
		{
			ImGui::SetNextWindowPos(ImVec2(0, 0));
			ImGui::Begin("Loading...", nullptr,
			             ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoInputs | ImGuiWindowFlags_NoNavInputs);
			ImGui::ProgressBar(1.0f - float(num) / cachedMaterials_.size(), ImVec2(ImGui::GetIO().DisplaySize.x, 32));
			ImGui::End();
		}

		if (showPerfStats_)
		{
			showTimeGPU();
		}
	}

	camera_.update(delta, mousePos_, mousePressed_);

	timestampBeginRendering = glfwGetTime();

	const float fov = float(45.0f * (M_PI / 180.0f));
	const float aspectRatio = (float)width_ / (float)height_;
	const glm::mat4 proj = glm::perspective(fov, aspectRatio, 0.5f, 500.0f);
	const glm::mat4 view = camera_.getViewMatrix();

	// Compute cascade split depths and light matrices
	const float nearClip = 0.5f;
	const float farClip = 500.0f;
	const float clipRange = farClip - nearClip;
	const glm::vec3 lightDir = kLightDir;
	const glm::mat4 invCam = glm::inverse(proj * view);
	const glm::mat4 scaleBias(0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.5, 0.5, 0.0, 1.0);

	float cascadeSplits[kNumCascades];
	{
		const float ratio = farClip / nearClip;
		for (uint32_t i = 0; i < kNumCascades; i++)
		{
			float p = (i + 1) / float(kNumCascades);
			float logSplit = nearClip * std::pow(ratio, p);
			float uniformSplit = nearClip + clipRange * p;
			float d = kCascadeSplitLambda * (logSplit - uniformSplit) + uniformSplit;
			cascadeSplits[i] = (d - nearClip) / clipRange;
		}
	}

	glm::mat4 shadowProjs[kNumCascades];
	glm::mat4 shadowViews[kNumCascades];

	float lastSplitDist = 0.0f;
	for (uint32_t i = 0; i < kNumCascades; i++)
	{
		float splitDist = cascadeSplits[i];

		glm::vec3 frustumCorners[8] = {
			{ -1.0f,  1.0f, 0.0f }, {  1.0f,  1.0f, 0.0f },
			{  1.0f, -1.0f, 0.0f }, { -1.0f, -1.0f, 0.0f },
			{ -1.0f,  1.0f, 1.0f }, {  1.0f,  1.0f, 1.0f },
			{  1.0f, -1.0f, 1.0f }, { -1.0f, -1.0f, 1.0f },
		};
		for (uint32_t j = 0; j < 8; j++)
		{
			glm::vec4 c = invCam * glm::vec4(frustumCorners[j], 1.0f);
			frustumCorners[j] = glm::vec3(c) / c.w;
		}
		for (uint32_t j = 0; j < 4; j++)
		{
			glm::vec3 dist = frustumCorners[j + 4] - frustumCorners[j];
			frustumCorners[j + 4] = frustumCorners[j] + dist * splitDist;
			frustumCorners[j] = frustumCorners[j] + dist * lastSplitDist;
		}

		glm::vec3 center(0.0f);
		for (uint32_t j = 0; j < 8; j++)
			center += frustumCorners[j];
		center /= 8.0f;

		float radius = 0.0f;
		for (uint32_t j = 0; j < 8; j++)
			radius = glm::max(radius, glm::length(frustumCorners[j] - center));
		radius = std::ceil(radius * 16.0f) / 16.0f;

		shadowViews[i] = glm::lookAt(center - lightDir * radius, center, glm::vec3(0.0f, 1.0f, 0.0f));
		shadowProjs[i] = glm::ortho(-radius, radius, -radius, radius, 0.0f, 2.0f * radius);

		perFrame_.cascadeLightMatrices[i] = scaleBias * shadowProjs[i] * shadowViews[i];
		perFrame_.cascadeSplitDepths[i] = -(nearClip + splitDist * clipRange);
		perFrame_.texShadow[i] = shadowCascadeTextures_[i].index();

		lastSplitDist = splitDist;
	}

	perFrame_.proj = proj;
	perFrame_.view = view;
	perFrame_.sampler = sampler_.index();
	perFrame_.samplerShadow = samplerShadow_.index();
	perFrame_.ambientColor = kAmbientColor;

	const glm::mat4 modelMatrix = glm::scale(glm::mat4(1.0f), glm::vec3(0.05f));
	const UniformsPerObject perObject = {
		.model = modelMatrix,
		.normal = glm::transpose(glm::inverse(modelMatrix)),
	};

	lvk::ICommandBuffer& buffer = ctx_->acquireCommandBuffer();

	processLoadedMaterials(buffer);

	buffer.cmdUpdateBuffer(ubPerFrame_, 0, sizeof(perFrame_), &perFrame_);
	buffer.cmdUpdateBuffer(ubPerObject_, 0, sizeof(perObject), &perObject);

	// Pass 1: shadows (one sub-pass per cascade)
	{
		struct
		{
			uint64_t perFrame;
			uint64_t perObject;
		} bindings = {
			.perFrame = ctx_->gpuAddress(ubPerFrameShadow_),
			.perObject = ctx_->gpuAddress(ubPerObject_),
		};

		for (uint32_t i = 0; i < kNumCascades; i++)
		{
			const UniformsPerFrameShadow perFrameShadow{
				.proj = shadowProjs[i],
				.view = shadowViews[i],
			};
			buffer.cmdUpdateBuffer(ubPerFrameShadow_, 0, sizeof(perFrameShadow), &perFrameShadow);
			buffer.cmdBeginRendering(renderPassShadow_, fbShadowCascades_[i]);
			{
				buffer.cmdBindRenderPipeline(renderPipelineState_Shadow_);
				buffer.cmdPushDebugGroupLabel("Render Shadows", 0xff0000ff);
				buffer.cmdBindDepthState(depthState_);
				buffer.cmdBindVertexBuffer(0, vb0_, 0);
				buffer.cmdPushConstants(bindings);
				buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
				buffer.cmdDrawIndexed((uint32_t)indexData_.size());
				buffer.cmdPopDebugGroupLabel();
			}
			buffer.cmdEndRendering();
			buffer.cmdTransitionToShaderReadOnly({ shadowCascadeTextures_[i] }, {});
		}
	}

#define GPU_TIMESTAMP(ts) buffer.cmdWriteTimestamp(queryPoolTimestamps_, ts);

	// Pass 2: mesh
	{
		buffer.cmdResetQueryPool(queryPoolTimestamps_, 0, GPUTimestamp_NUM_TIMESTAMPS);
		GPU_TIMESTAMP(GPUTimestamp_BeginSceneRendering);

		buffer.cmdBeginRendering(renderPassOffscreen_, fbOffscreen_);
		{
			buffer.cmdBindRenderPipeline(drawNormals_ ? renderPipelineState_MeshNormals_ : renderPipelineState_Mesh_);
			buffer.cmdPushDebugGroupLabel("Render Mesh", 0xff0000ff);
			buffer.cmdBindDepthState(depthState_);
			buffer.cmdBindVertexBuffer(0, vb0_, 0);

			struct
			{
				uint64_t perFrame;
				uint64_t perObject;
				uint64_t materials;
			} bindings = {
				.perFrame = ctx_->gpuAddress(ubPerFrame_),
				.perObject = ctx_->gpuAddress(ubPerObject_),
				.materials = ctx_->gpuAddress(sbMaterials_),
			};

			buffer.cmdPushConstants(bindings);
			buffer.cmdBindIndexBuffer(ib0_, lvk::IndexFormat_UI32);
			buffer.cmdDrawIndexed((uint32_t)indexData_.size());
			if (enableWireframe_)
			{
				buffer.cmdBindRenderPipeline(renderPipelineState_MeshWireframe_);
				buffer.cmdDrawIndexed((uint32_t)indexData_.size());
			}
			buffer.cmdPopDebugGroupLabel();
		}
		buffer.cmdEndRendering();

		GPU_TIMESTAMP(GPUTimestamp_EndSceneRendering);
	}

	// Pass 3: fullscreen blit to swapchain
	{
		GPU_TIMESTAMP(GPUTimestamp_BeginPresent);

		const lvk::TextureHandle tex = fbOffscreen_.color[0].texture;
		buffer.cmdBeginRendering(renderPassMain_, fbMain_, { .sampledImages = { tex } });
		{
			buffer.cmdBindRenderPipeline(renderPipelineState_Fullscreen_);
			buffer.cmdPushDebugGroupLabel("Swapchain Output", 0xff0000ff);
			buffer.cmdBindDepthState(depthState_);

			struct
			{
				uint32_t texture;
			} bindings = {
				.texture = tex.index(),
			};

			buffer.cmdPushConstants(bindings);
			buffer.cmdDraw(3);
			buffer.cmdPopDebugGroupLabel();

			imgui_->endFrame(buffer);
		}
		buffer.cmdEndRendering();

		GPU_TIMESTAMP(GPUTimestamp_EndPresent);
	}

#undef GPU_TIMESTAMP

	ctx_->submit(buffer, fbMain_.color[0].texture);

	timestampEndRendering = glfwGetTime();

	if (showPerfStats_)
	{
		ctx_->getQueryPoolResults(queryPoolTimestamps_, 0, LVK_ARRAY_NUM_ELEMENTS(pipelineTimestamps),
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

	lvk::LVKwindow* window = lvk::initWindow("Continuous CSM", width_, height_);
	ctx_ = lvk::createVulkanContextWithSwapchain(window, width_, height_,
	                                             {
	                                                 .enableValidation = kEnableValidationLayers,
	                                             },
	                                             kPreferIntegratedGPU ? lvk::HWDeviceType_Integrated
	                                                                  : lvk::HWDeviceType_Discrete);
	if (!ctx_)
	{
		return EXIT_FAILURE;
	}

	if (!init(window))
	{
		return EXIT_FAILURE;
	}

	double prevTime = glfwGetTime();

	glfwSetFramebufferSizeCallback(window,
	                               [](GLFWwindow*, int32_t w, int32_t h)
	                               {
		                               width_ = w;
		                               height_ = h;
		                               resize();
	                               });

	glfwSetCursorPosCallback(window,
	                         [](GLFWwindow* w, double x, double y)
	                         {
		                         int32_t width, height;
		                         glfwGetFramebufferSize(w, &width, &height);
		                         if (width && height)
		                         {
			                         mousePos_ = glm::vec2(x / width, 1.0f - y / height);
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
				    mousePressed_ = (action == GLFW_PRESS);
				    glfwSetInputMode(w, GLFW_CURSOR, mousePressed_ ? GLFW_CURSOR_DISABLED : GLFW_CURSOR_NORMAL);
			    }
		    }
		    else
		    {
			    mousePressed_ = false;
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
			                       loaderShouldExit_.store(true, std::memory_order_release);
			                       glfwSetWindowShouldClose(w, GLFW_TRUE);
		                       }
		                       if (key == GLFW_KEY_N && pressed)
		                       {
			                       drawNormals_ = !drawNormals_;
		                       }
		                       if (key == GLFW_KEY_T && pressed)
		                       {
			                       enableWireframe_ = !enableWireframe_;
		                       }
		                       if (key == GLFW_KEY_P && pressed)
		                       {
			                       showPerfStats_ = !showPerfStats_;
		                       }
		                       if (key == GLFW_KEY_W)
		                       {
			                       camera_.movement_.forward_ = pressed;
		                       }
		                       if (key == GLFW_KEY_S)
		                       {
			                       camera_.movement_.backward_ = pressed;
		                       }
		                       if (key == GLFW_KEY_A)
		                       {
			                       camera_.movement_.left_ = pressed;
		                       }
		                       if (key == GLFW_KEY_D)
		                       {
			                       camera_.movement_.right_ = pressed;
		                       }
		                       if (key == GLFW_KEY_Q)
		                       {
			                       camera_.movement_.up_ = pressed;
		                       }
		                       if (key == GLFW_KEY_E)
		                       {
			                       camera_.movement_.down_ = pressed;
		                       }
		                       if (mods & GLFW_MOD_SHIFT)
		                       {
			                       camera_.movement_.fastSpeed_ = pressed;
		                       }
		                       if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT)
		                       {
			                       camera_.movement_.fastSpeed_ = pressed;
		                       }
		                       if (key == GLFW_KEY_SPACE)
		                       {
			                       camera_.setUpVector(glm::vec3(0.0f, 1.0f, 0.0f));
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

		if (!width_ || !height_)
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
