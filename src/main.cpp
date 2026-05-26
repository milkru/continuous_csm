#include <GLFW/glfw3.h>
#include <lvk/LVK.h>

const char* codeVS = R"(
	#version 460
	layout(location = 0) out vec3 color;
	const vec2 pos[3] = vec2[3](
		vec2(-0.6, -0.4),
		vec2( 0.6, -0.4),
		vec2( 0.0,  0.6)
	);
	const vec3 col[3] = vec3[3](
		vec3(1.0, 0.0, 0.0),
		vec3(0.0, 1.0, 0.0),
		vec3(0.0, 0.0, 1.0)
	);
	void main()
	{
		gl_Position = vec4(pos[gl_VertexIndex], 0.0, 1.0);
		color = col[gl_VertexIndex];
	}
)";

const char* codeFS = R"(
	#version 460
	layout(location = 0) in vec3 color;
	layout(location = 0) out vec4 out_FragColor;
	void main()
	{
		out_FragColor = vec4(color, 1.0);
	}
)";

int main(int argc, char* argv[])
{
	int32_t width = 800;
	int32_t height = 600;

	lvk::LVKwindow* window = lvk::initWindow("Continuous CSM", width, height, /*resizable=*/true);
	std::unique_ptr<lvk::IContext> ctx = lvk::createVulkanContextWithSwapchain(window, width, height, {});

	lvk::Holder<lvk::ShaderModuleHandle> vert = ctx->createShaderModule({codeVS, lvk::Stage_Vert, "Shader: vert"});
	lvk::Holder<lvk::ShaderModuleHandle> frag = ctx->createShaderModule({codeFS, lvk::Stage_Frag, "Shader: frag"});

	lvk::Holder<lvk::RenderPipelineHandle> pipeline = ctx->createRenderPipeline(
	    {
	        .smVert = vert,
	        .smFrag = frag,
	        .color = {{.format = ctx->getSwapchainFormat()}},
	    },
	    nullptr);
	LVK_ASSERT(pipeline.valid());

	glfwSetWindowUserPointer(window, ctx.get());
	glfwSetKeyCallback(window,
	                   [](GLFWwindow* w, int key, int /*scancode*/, int action, int /*mods*/)
	                   {
		                   if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		                   {
			                   glfwSetWindowShouldClose(w, GLFW_TRUE);
		                   }
	                   });
	glfwSetFramebufferSizeCallback(
	    window, [](GLFWwindow* w, int newWidth, int newHeight)
	    { static_cast<lvk::IContext*>(glfwGetWindowUserPointer(w))->recreateSwapchain(newWidth, newHeight); });

	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();
		glfwGetFramebufferSize(window, &width, &height);
		if (!width || !height)
		{
			continue;
		}

		lvk::ICommandBuffer& cmd = ctx->acquireCommandBuffer();
		cmd.cmdBeginRendering({.color = {{.loadOp = lvk::LoadOp_Clear, .clearColor = {0.1f, 0.1f, 0.1f, 1.0f}}}},
		                      {.color = {{.texture = ctx->getCurrentSwapchainTexture()}}});
		cmd.cmdBindRenderPipeline(pipeline);
		cmd.cmdDraw(3);
		cmd.cmdEndRendering();
		ctx->submit(cmd, ctx->getCurrentSwapchainTexture());
	}

	ctx = nullptr;
	glfwDestroyWindow(window);
	glfwTerminate();

	return 0;
}
