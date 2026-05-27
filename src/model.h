#pragma once

#define MAX_MATERIAL_NAME 128

struct VertexData
{
	glm::vec3 position;
	uint32_t uv;
	uint16_t normal;
	uint16_t mtlIndex;
};

static_assert(sizeof(VertexData) == 5 * sizeof(uint32_t));

struct CachedMaterial
{
	char name[MAX_MATERIAL_NAME] = {};
	glm::vec3 ambient = glm::vec3(0.0f);
	glm::vec3 diffuse = glm::vec3(0.0f);
	char ambient_texname[MAX_MATERIAL_NAME] = {};
	char diffuse_texname[MAX_MATERIAL_NAME] = {};
	char alpha_texname[MAX_MATERIAL_NAME] = {};
};

struct GPUMaterial
{
	glm::vec4 ambient = glm::vec4(0.0f);
	glm::vec4 diffuse = glm::vec4(0.0f);
	uint32_t texAmbient = 0;
	uint32_t texDiffuse = 0;
	uint32_t texAlpha = 0;
	uint32_t padding[1];
};

static_assert(sizeof(GPUMaterial) % 16 == 0);

struct MaterialTextures
{
	lvk::TextureHandle ambient;
	lvk::TextureHandle diffuse;
	lvk::TextureHandle alpha;
};

struct LoadedImage
{
	uint32_t w = 0;
	uint32_t h = 0;
	uint32_t channels = 0;
	uint8_t* pixels = nullptr;
	std::string debugName;
};

struct LoadedMaterial
{
	size_t idx = 0;
	LoadedImage ambient;
	LoadedImage diffuse;
	LoadedImage alpha;
};

extern std::vector<uint32_t> indexData_;
extern std::vector<CachedMaterial> cachedMaterials_;
extern std::vector<MaterialTextures> textures_;
extern std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> texturesCache_;
extern std::atomic<uint32_t> remainingMaterialsToLoad_;
extern std::atomic<bool> loaderShouldExit_;
extern std::unique_ptr<tf::Executor> loaderPool_;

bool initModel();
void loadMaterials();
void processLoadedMaterials(lvk::ICommandBuffer&);
