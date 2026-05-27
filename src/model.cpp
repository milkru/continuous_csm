#include <algorithm>
#include <cassert>
#include <filesystem>
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

extern std::unique_ptr<lvk::IContext> ctx_;
extern std::string folderContentRoot;
extern lvk::Holder<lvk::TextureHandle> textureDummyWhite_;
extern lvk::Holder<lvk::BufferHandle> sbMaterials_;

constexpr uint32_t kMeshCacheVersion = 0xC0DE000A;

// mesh globals
std::vector<VertexData> vertexData_;
std::vector<uint32_t> indexData_;
std::vector<CachedMaterial> cachedMaterials_;
std::vector<GPUMaterial> materials_;
std::vector<MaterialTextures> textures_;
std::vector<LoadedMaterial> loadedMaterials_;
std::unordered_map<std::string, LoadedImage> imagesCache_;
std::unordered_map<std::string, lvk::Holder<lvk::TextureHandle>> texturesCache_;
std::mutex imagesCacheMutex_;
std::mutex loadedMaterialsMutex_;
std::atomic<bool> loaderShouldExit_ = false;
std::atomic<uint32_t> remainingMaterialsToLoad_ = 0;
std::unique_ptr<tf::Executor> loaderPool_ =
    std::make_unique<tf::Executor>(std::max(2u, std::thread::hardware_concurrency() / 2));

// forward declarations used within mesh.cpp
extern lvk::Holder<lvk::BufferHandle> vb0_;
extern lvk::Holder<lvk::BufferHandle> ib0_;

static bool endsWith(const std::string& str, const std::string& suffix)
{
	return str.size() >= suffix.size() && 0 == str.compare(str.size() - suffix.size(), suffix.size(), suffix);
}

static void stringReplaceAll(std::string& s, const std::string& searchString, const std::string& replaceString)
{
	size_t pos = 0;
	while ((pos = s.find(searchString, pos)) != std::string::npos)
	{
		s.replace(pos, searchString.length(), replaceString);
		pos += replaceString.length();
	}
}

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

	vertexData_.reserve(vertexCount);

	uint32_t vertexIndex = 0;
	for (uint32_t face = 0; face < mesh->face_count; face++)
	{
		for (uint32_t v = 0; v < mesh->face_vertices[face]; v++)
		{
			LVK_ASSERT(v < 3);
			const fastObjIndex gi = mesh->indices[vertexIndex++];
			const float* p = &mesh->positions[gi.p * 3];
			const float* n = &mesh->normals[gi.n * 3];
			const float* t = &mesh->texcoords[gi.t * 2];
			vertexData_.push_back({
			    .position = glm::vec3(p[0], p[1], p[2]),
			    .uv = glm::packHalf2x16(glm::vec2(t[0], t[1])),
			    .normal = packOctahedral16(glm::vec3(n[0], n[1], n[2])),
			    .mtlIndex = (uint16_t)mesh->face_materials[face],
			});
		}
	}

	{
		const size_t indexCount = vertexData_.size();
		std::vector<uint32_t> remap(indexCount);
		const size_t vertexCountOpt = meshopt_generateVertexRemap(remap.data(), nullptr, indexCount, vertexData_.data(),
		                                                          indexCount, sizeof(VertexData));
		std::vector<VertexData> remappedVertices;
		indexData_.resize(indexCount);
		remappedVertices.resize(vertexCountOpt);
		meshopt_remapIndexBuffer(indexData_.data(), nullptr, indexCount, &remap[0]);
		meshopt_remapVertexBuffer(remappedVertices.data(), vertexData_.data(), indexCount, sizeof(VertexData),
		                          remap.data());
		vertexData_ = remappedVertices;
		meshopt_optimizeVertexCache(indexData_.data(), indexData_.data(), indexCount, vertexCountOpt);
		meshopt_optimizeOverdraw(indexData_.data(), indexData_.data(), indexCount, &vertexData_[0].position.x,
		                         vertexCountOpt, sizeof(VertexData), 1.05f);
		meshopt_optimizeVertexFetch(vertexData_.data(), indexData_.data(), indexCount, vertexData_.data(),
		                            vertexCountOpt, sizeof(VertexData));
	}

	for (uint32_t mtlIdx = 0; mtlIdx != mesh->material_count; mtlIdx++)
	{
		const fastObjMaterial& m = mesh->materials[mtlIdx];
		CachedMaterial mtl;
		mtl.ambient = glm::vec3(m.Ka[0], m.Ka[1], m.Ka[2]);
		mtl.diffuse = glm::vec3(m.Kd[0], m.Kd[1], m.Kd[2]);
		LVK_ASSERT(strlen(m.name) < MAX_MATERIAL_NAME);
		strcat(mtl.name, m.name);
		strcat(mtl.ambient_texname, normalizeTextureName(mesh->textures[m.map_Ka].name).c_str());
		strcat(mtl.diffuse_texname, normalizeTextureName(mesh->textures[m.map_Kd].name).c_str());
		strcat(mtl.alpha_texname, normalizeTextureName(mesh->textures[m.map_d].name).c_str());
		cachedMaterials_.push_back(mtl);
	}

	LLOGL("Caching mesh...\n");

	fast_obj_destroy(mesh);
	mesh = nullptr;

	FILE* cacheFile = fopen(cacheFileName, "wb");
	if (!cacheFile)
	{
		return false;
	}
	const uint32_t numMaterials = (uint32_t)cachedMaterials_.size();
	const uint32_t numVertices = (uint32_t)vertexData_.size();
	const uint32_t numIndices = (uint32_t)indexData_.size();
	fwrite(&kMeshCacheVersion, sizeof(kMeshCacheVersion), 1, cacheFile);
	fwrite(&numMaterials, sizeof(numMaterials), 1, cacheFile);
	fwrite(&numVertices, sizeof(numVertices), 1, cacheFile);
	fwrite(&numIndices, sizeof(numIndices), 1, cacheFile);
	fwrite(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile);
	fwrite(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile);
	fwrite(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile);
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
	cachedMaterials_.resize(numMaterials);
	vertexData_.resize(numVertices);
	indexData_.resize(numIndices);
	CHECK_READ(numMaterials, fread(cachedMaterials_.data(), sizeof(CachedMaterial), numMaterials, cacheFile));
	CHECK_READ(numVertices, fread(vertexData_.data(), sizeof(VertexData), numVertices, cacheFile));
	CHECK_READ(numIndices, fread(indexData_.data(), sizeof(uint32_t), numIndices, cacheFile));
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

	for (const auto& mtl : cachedMaterials_)
	{
		materials_.push_back(GPUMaterial{ glm::vec4(mtl.ambient, 1.0f), glm::vec4(mtl.diffuse, 1.0f),
		                                  textureDummyWhite_.index(), textureDummyWhite_.index() });
	}

	sbMaterials_ = ctx_->createBuffer(
	    {
	        .usage = lvk::BufferUsageBits_Storage,
	        .storage = lvk::StorageType_Device,
	        .size = sizeof(GPUMaterial) * materials_.size(),
	        .data = materials_.data(),
	        .debugName = "Buffer: materials",
	    },
	    nullptr);

	vb0_ = ctx_->createBuffer(
	    {
	        .usage = lvk::BufferUsageBits_Vertex,
	        .storage = lvk::StorageType_Device,
	        .size = sizeof(VertexData) * vertexData_.size(),
	        .data = vertexData_.data(),
	        .debugName = "Buffer: vertex",
	    },
	    nullptr);
	ib0_ = ctx_->createBuffer(
	    {
	        .usage = lvk::BufferUsageBits_Index,
	        .storage = lvk::StorageType_Device,
	        .size = sizeof(uint32_t) * indexData_.size(),
	        .data = indexData_.data(),
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
		std::lock_guard lock(imagesCacheMutex_);
		const auto it = imagesCache_.find(debugName);
		if (it != imagesCache_.end())
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

	std::lock_guard lock(imagesCacheMutex_);
	imagesCache_[fileName] = img;
	return img;
}

void loadMaterial(size_t i)
{
	static const std::string pathPrefix = folderContentRoot + "src/bistro/Exterior/";

	remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);

#define LOAD_TEX(result, tex, channels)                                                                                \
	const LoadedImage result = std::string(cachedMaterials_[i].tex).empty()                                            \
	                               ? LoadedImage()                                                                     \
	                               : loadImage((pathPrefix + cachedMaterials_[i].tex).c_str(), channels);              \
	if (loaderShouldExit_.load(std::memory_order_acquire))                                                             \
	{                                                                                                                  \
		return;                                                                                                        \
	}

	LOAD_TEX(ambient, ambient_texname, 4);
	LOAD_TEX(diffuse, diffuse_texname, 4);
	LOAD_TEX(alpha, alpha_texname, 1);

#undef LOAD_TEX

	const LoadedMaterial mtl{ i, ambient, diffuse, alpha };

	if (!mtl.ambient.pixels && !mtl.diffuse.pixels)
	{
		materials_[i].texDiffuse = 0;
	}
	else
	{
		std::lock_guard guard(loadedMaterialsMutex_);
		loadedMaterials_.push_back(mtl);
		remainingMaterialsToLoad_.fetch_add(1u, std::memory_order_release);
	}
}

void loadMaterials()
{
	stbi_set_flip_vertically_on_load(1);
	remainingMaterialsToLoad_ = (uint32_t)cachedMaterials_.size();
	textures_.resize(cachedMaterials_.size());
	for (size_t i = 0; i != cachedMaterials_.size(); i++)
	{
		loaderPool_->silent_async([i]() { loadMaterial(i); });
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

	const auto it = texturesCache_.find(img.debugName);
	if (it != texturesCache_.end())
	{
		return it->second;
	}

	lvk::Holder<lvk::TextureHandle> tex = ctx_->createTexture({
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

	const lvk::TextureHandle handle = tex;
	texturesCache_[img.debugName] = std::move(tex);
	return handle;
}

void processLoadedMaterials(lvk::ICommandBuffer& buffer)
{
	LoadedMaterial mtl;

	{
		std::lock_guard guard(loadedMaterialsMutex_);
		if (loadedMaterials_.empty())
		{
			return;
		}
		mtl = loadedMaterials_.back();
		loadedMaterials_.pop_back();
		remainingMaterialsToLoad_.fetch_sub(1u, std::memory_order_release);
	}

	{
		MaterialTextures tex;
		tex.ambient = createTexture(mtl.ambient);
		tex.diffuse = createTexture(mtl.diffuse);
		tex.alpha = createTexture(mtl.alpha);

		materials_[mtl.idx].texAmbient = tex.ambient.index();
		materials_[mtl.idx].texDiffuse = tex.diffuse.index();
		materials_[mtl.idx].texAlpha = tex.alpha.index();
		textures_[mtl.idx] = std::move(tex);
	}

	LVK_ASSERT(materials_[mtl.idx].texAmbient >= 0);
	LVK_ASSERT(materials_[mtl.idx].texDiffuse >= 0);
	LVK_ASSERT(materials_[mtl.idx].texAlpha >= 0);
	buffer.cmdUpdateBuffer(sbMaterials_, 0, sizeof(GPUMaterial) * materials_.size(), materials_.data());
}
