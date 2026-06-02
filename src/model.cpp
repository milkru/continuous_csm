#include <algorithm>
#include <mutex>
#include <stdio.h>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <glm/glm.hpp>
#include <lvk/LVK.h>

#define FAST_OBJ_IMPLEMENTATION
#include <fast_obj.h>
#include <meshoptimizer.h>

#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>

#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include <stb_image_resize2.h>
#include <taskflow/taskflow.hpp>

#include "model.h"
#include "utils.h"

extern std::unique_ptr<lvk::IContext> context;
extern std::string folderContentRoot;
extern lvk::Holder<lvk::TextureHandle> textureDummyWhite;
extern lvk::Holder<lvk::BufferHandle> materialsBuffer;

constexpr uint32_t kMeshCacheVersion = 0xC0DE000A;

// mesh globals
std::vector<VertexData> vertexData;
std::vector<uint32_t> indexData;
glm::vec3 sceneAABBMin(0.0f);
glm::vec3 sceneAABBMax(0.0f);
std::vector<CachedMaterial> cachedMaterials;
std::vector<GPUMaterial> materials;
std::vector<MaterialTextures> textures;
std::vector<LoadedMaterial> loadedMaterials;
std::unordered_map<std::string, LoadedImage> imagesCache;
std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> texturesCache;
std::mutex imagesCacheMutex;
std::mutex loadedMaterialsMutex;
std::atomic<bool> loaderShouldExit = false;
std::atomic<uint32_t> remainingMaterialsToLoad = 0;
std::unique_ptr<tf::Executor> loaderPool =
    std::make_unique<tf::Executor>(std::max(2u, std::thread::hardware_concurrency() / 2));

// forward declarations used within mesh.cpp
extern lvk::Holder<lvk::BufferHandle> vertexBuffer;
extern lvk::Holder<lvk::BufferHandle> indexBuffer;

std::string normalizeTextureName(const char* n)
{
	if (!n)
	{
		return std::string();
	}
	LVK_ASSERT(strlen(n) < MAX_MATERIAL_NAME);
	std::string name(n);
	return name;
}

bool loadAndCache(const char* cacheFileName)
{
	LLOGL("Loading `exterior.obj`... It can take a while in debug builds...\n");

	fastObjMesh* mesh = fast_obj_read((folderContentRoot + "src/bistro/Exterior/exterior.obj").c_str());

	if (!LVK_VERIFY(mesh))
	{
		LVK_ASSERT_MSG(false, "Did you read the tutorial at the top of this file?");
		return false;
	}

	uint32_t vertexCount = 0;
	for (uint32_t i = 0; i < mesh->face_count; ++i)
	{
		vertexCount += mesh->face_vertices[i];
	}

	vertexData.reserve(vertexCount);

	uint32_t vertexIndex = 0;
	for (uint32_t face = 0; face < mesh->face_count; face++)
	{
		for (uint32_t v = 0; v < mesh->face_vertices[face]; v++)
		{
			LVK_ASSERT(v < 3);
			const fastObjIndex objIndex = mesh->indices[vertexIndex++];
			const float* positionPtr = &mesh->positions[objIndex.p * 3];
			const float* normalPtr = &mesh->normals[objIndex.n * 3];
			const float* texcoordPtr = &mesh->texcoords[objIndex.t * 2];
			vertexData.push_back({
			    .position = glm::vec3(positionPtr[0], positionPtr[1], positionPtr[2]),
			    .uv = glm::packHalf2x16(glm::vec2(texcoordPtr[0], texcoordPtr[1])),
			    .normal = packOctahedral16(glm::vec3(normalPtr[0], normalPtr[1], normalPtr[2])),
			    .mtlIndex = (uint16_t)mesh->face_materials[face],
			});
		}
	}

	{
		const size_t indexCount = vertexData.size();
		std::vector<uint32_t> remap(indexCount);
		const size_t vertexCountOpt = meshopt_generateVertexRemap(remap.data(), nullptr, indexCount, vertexData.data(),
		                                                          indexCount, sizeof(VertexData));
		std::vector<VertexData> remappedVertices;
		indexData.resize(indexCount);
		remappedVertices.resize(vertexCountOpt);
		meshopt_remapIndexBuffer(indexData.data(), nullptr, indexCount, &remap[0]);
		meshopt_remapVertexBuffer(remappedVertices.data(), vertexData.data(), indexCount, sizeof(VertexData),
		                          remap.data());
		vertexData = remappedVertices;
		meshopt_optimizeVertexCache(indexData.data(), indexData.data(), indexCount, vertexCountOpt);
		meshopt_optimizeOverdraw(indexData.data(), indexData.data(), indexCount, &vertexData[0].position.x,
		                         vertexCountOpt, sizeof(VertexData), 1.05f);
		meshopt_optimizeVertexFetch(vertexData.data(), indexData.data(), indexCount, vertexData.data(), vertexCountOpt,
		                            sizeof(VertexData));
	}

	for (uint32_t materialIndex = 0; materialIndex != mesh->material_count; materialIndex++)
	{
		const fastObjMaterial& objMaterial = mesh->materials[materialIndex];
		CachedMaterial material;
		material.ambient = glm::vec3(objMaterial.Ka[0], objMaterial.Ka[1], objMaterial.Ka[2]);
		material.diffuse = glm::vec3(objMaterial.Kd[0], objMaterial.Kd[1], objMaterial.Kd[2]);
		LVK_ASSERT(strlen(objMaterial.name) < MAX_MATERIAL_NAME);
		strcat(material.name, objMaterial.name);
		strcat(material.ambient_texname, normalizeTextureName(mesh->textures[objMaterial.map_Ka].name).c_str());
		strcat(material.diffuse_texname, normalizeTextureName(mesh->textures[objMaterial.map_Kd].name).c_str());
		strcat(material.alpha_texname, normalizeTextureName(mesh->textures[objMaterial.map_d].name).c_str());
		cachedMaterials.push_back(material);
	}

	LLOGL("Caching mesh...\n");

	fast_obj_destroy(mesh);
	mesh = nullptr;

	FILE* cacheFile = fopen(cacheFileName, "wb");
	if (!cacheFile)
	{
		return false;
	}
	const uint32_t numMaterials = (uint32_t)cachedMaterials.size();
	const uint32_t numVertices = (uint32_t)vertexData.size();
	const uint32_t numIndices = (uint32_t)indexData.size();
	fwrite(&kMeshCacheVersion, sizeof(kMeshCacheVersion), 1, cacheFile);
	fwrite(&numMaterials, sizeof(numMaterials), 1, cacheFile);
	fwrite(&numVertices, sizeof(numVertices), 1, cacheFile);
	fwrite(&numIndices, sizeof(numIndices), 1, cacheFile);
	fwrite(cachedMaterials.data(), sizeof(CachedMaterial), numMaterials, cacheFile);
	fwrite(vertexData.data(), sizeof(VertexData), numVertices, cacheFile);
	fwrite(indexData.data(), sizeof(uint32_t), numIndices, cacheFile);
	return fclose(cacheFile) == 0;
}

bool loadFromCache(const char* cacheFileName)
{
	FILE* cacheFile = fopen(cacheFileName, "rb");
	if (!cacheFile)
	{
		return false;
	}
#define CHECK_READ(expected, read)                                                                                     \
	if ((read) != (expected))                                                                                          \
	{                                                                                                                  \
		fclose(cacheFile);                                                                                             \
		return false;                                                                                                  \
	}
	uint32_t versionProbe = 0;
	CHECK_READ(1, fread(&versionProbe, sizeof(versionProbe), 1, cacheFile));
	if (versionProbe != kMeshCacheVersion)
	{
		LLOGL("Cache file has wrong version id\n");
		fclose(cacheFile);
		return false;
	}
	uint32_t numMaterials = 0;
	uint32_t numVertices = 0;
	uint32_t numIndices = 0;
	CHECK_READ(1, fread(&numMaterials, sizeof(numMaterials), 1, cacheFile));
	CHECK_READ(1, fread(&numVertices, sizeof(numVertices), 1, cacheFile));
	CHECK_READ(1, fread(&numIndices, sizeof(numIndices), 1, cacheFile));
	cachedMaterials.resize(numMaterials);
	vertexData.resize(numVertices);
	indexData.resize(numIndices);
	CHECK_READ(numMaterials, fread(cachedMaterials.data(), sizeof(CachedMaterial), numMaterials, cacheFile));
	CHECK_READ(numVertices, fread(vertexData.data(), sizeof(VertexData), numVertices, cacheFile));
	CHECK_READ(numIndices, fread(indexData.data(), sizeof(uint32_t), numIndices, cacheFile));
#undef CHECK_READ
	fclose(cacheFile);
	return true;
}

bool initModel()
{
	const std::string cacheFileName = folderContentRoot + "cache.data";

	if (!loadFromCache(cacheFileName.c_str()))
	{
		if (!LVK_VERIFY(loadAndCache(cacheFileName.c_str())))
		{
			LVK_ASSERT_MSG(false, "Cannot load 3D model");
			return false;
		}
	}

	for (const auto& mtl : cachedMaterials)
	{
		materials.push_back(GPUMaterial{ glm::vec4(mtl.ambient, 1.0f), glm::vec4(mtl.diffuse, 1.0f),
		                                 textureDummyWhite.index(), textureDummyWhite.index() });
	}

	materialsBuffer = context->createBuffer(
	    {
	        .usage = lvk::BufferUsageBits_Storage,
	        .storage = lvk::StorageType_Device,
	        .size = sizeof(GPUMaterial) * materials.size(),
	        .data = materials.data(),
	        .debugName = "Buffer: materials",
	    },
	    nullptr);

	sceneAABBMin = glm::vec3(FLT_MAX);
	sceneAABBMax = glm::vec3(-FLT_MAX);
	for (const VertexData& v : vertexData)
	{
		sceneAABBMin = glm::min(sceneAABBMin, v.position);
		sceneAABBMax = glm::max(sceneAABBMax, v.position);
	}

	vertexBuffer = context->createBuffer(
	    {
	        .usage = lvk::BufferUsageBits_Vertex,
	        .storage = lvk::StorageType_Device,
	        .size = sizeof(VertexData) * vertexData.size(),
	        .data = vertexData.data(),
	        .debugName = "Buffer: vertex",
	    },
	    nullptr);
	indexBuffer = context->createBuffer(
	    {
	        .usage = lvk::BufferUsageBits_Index,
	        .storage = lvk::StorageType_Device,
	        .size = sizeof(uint32_t) * indexData.size(),
	        .data = indexData.data(),
	        .debugName = "Buffer: index",
	    },
	    nullptr);
	return true;
}

LoadedImage loadImage(const char* fileName, int32_t channels)
{
	if (!fileName || !*fileName)
	{
		return LoadedImage();
	}

	char debugStr[512] = { 0 };
	snprintf(debugStr, sizeof(debugStr) - 1, "%s (%i)", fileName, channels);
	const std::string debugName(debugStr);

	{
		std::lock_guard lock(imagesCacheMutex);
		const auto it = imagesCache.find(debugName);
		if (it != imagesCache.end())
		{
			LVK_ASSERT(channels == (int32_t)it->second.channels);
			return it->second;
		}
	}

	int32_t w, h;
	uint8_t* pixels = stbi_load(fileName, &w, &h, nullptr, channels);

	const LoadedImage img = {
		.w = (uint32_t)w,
		.h = (uint32_t)h,
		.channels = (uint32_t)channels,
		.pixels = pixels,
		.debugName = debugName,
	};

	std::lock_guard lock(imagesCacheMutex);
	imagesCache[fileName] = img;
	return img;
}

void loadMaterial(size_t i)
{
	static const std::string pathPrefix = folderContentRoot + "src/bistro/Exterior/";

	remainingMaterialsToLoad.fetch_sub(1u, std::memory_order_release);

#define LOAD_TEX(result, tex, channels)                                                                                \
	const LoadedImage result = std::string(cachedMaterials[i].tex).empty()                                             \
	                               ? LoadedImage()                                                                     \
	                               : loadImage((pathPrefix + cachedMaterials[i].tex).c_str(), channels);               \
	if (loaderShouldExit.load(std::memory_order_acquire))                                                              \
	{                                                                                                                  \
		return;                                                                                                        \
	}

	LOAD_TEX(ambient, ambient_texname, 4);
	LOAD_TEX(diffuse, diffuse_texname, 4);
	LOAD_TEX(alpha, alpha_texname, 1);

#undef LOAD_TEX

	const LoadedMaterial material{ i, ambient, diffuse, alpha };

	if (!material.ambient.pixels && !material.diffuse.pixels)
	{
		materials[i].texDiffuse = 0;
	}
	else
	{
		std::lock_guard guard(loadedMaterialsMutex);
		loadedMaterials.push_back(material);
		remainingMaterialsToLoad.fetch_add(1u, std::memory_order_release);
	}
}

void loadMaterials()
{
	stbi_set_flip_vertically_on_load(1);
	remainingMaterialsToLoad = (uint32_t)cachedMaterials.size();
	textures.resize(cachedMaterials.size());
	for (size_t i = 0; i != cachedMaterials.size(); i++)
	{
		loaderPool->silent_async([i]() { loadMaterial(i); });
	}
}

lvk::Format formatFromChannels(uint32_t channels)
{
	if (channels == 1)
	{
		return lvk::Format_R_UN8;
	}
	if (channels == 4)
	{
		return lvk::Format_RGBA_UN8;
	}
	return lvk::Format_Invalid;
}

lvk::TextureHandle createTexture(const LoadedImage& img)
{
	if (!img.pixels)
	{
		return {};
	}

	const auto it = texturesCache.find(img.debugName);
	if (it != texturesCache.end())
	{
		return it->second;
	}

	lvk::Holder<lvk::TextureHandle> texture = context->createTexture({
	    .type = lvk::TextureType_2D,
	    .format = formatFromChannels(img.channels),
	    .dimensions = { img.w, img.h },
	    .usage = lvk::TextureUsageBits_Sampled,
	    .numMipLevels = lvk::calcNumMipLevels(img.w, img.h),
	    .components = (img.channels == 1)
	                      ? lvk::ComponentMapping{ lvk::Swizzle_R, lvk::Swizzle_R, lvk::Swizzle_R, lvk::Swizzle_R }
	                      : lvk::ComponentMapping{},
	    .data = img.pixels,
	    .dataNumMipLevels = 1u,
	    .generateMipmaps = true,
	    .debugName = img.debugName.c_str(),
	});

	const lvk::TextureHandle handle = texture;
	texturesCache[img.debugName] = std::move(texture);
	return handle;
}

void processLoadedMaterials(lvk::ICommandBuffer& buffer)
{
	LoadedMaterial material;

	{
		std::lock_guard guard(loadedMaterialsMutex);
		if (loadedMaterials.empty())
		{
			return;
		}
		material = loadedMaterials.back();
		loadedMaterials.pop_back();
		remainingMaterialsToLoad.fetch_sub(1u, std::memory_order_release);
	}

	{
		MaterialTextures materialTextures;
		materialTextures.ambient = createTexture(material.ambient);
		materialTextures.diffuse = createTexture(material.diffuse);
		materialTextures.alpha = createTexture(material.alpha);

		materials[material.idx].texAmbient = materialTextures.ambient.index();
		materials[material.idx].texDiffuse = materialTextures.diffuse.index();
		materials[material.idx].texAlpha = materialTextures.alpha.index();
		textures[material.idx] = std::move(materialTextures);
	}

	LVK_ASSERT(materials[material.idx].texAmbient >= 0);
	LVK_ASSERT(materials[material.idx].texDiffuse >= 0);
	LVK_ASSERT(materials[material.idx].texAlpha >= 0);
	buffer.cmdUpdateBuffer(materialsBuffer, 0, sizeof(GPUMaterial) * materials.size(), materials.data());
}
