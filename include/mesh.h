#pragma once
#ifndef __MESH_H
#define __MESH_H
#include <string>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>
#include "triangle.h"

namespace fs = std::filesystem;

class Mesh : public Hittable {
public:
	Mesh();
	void loadMesh(aiMesh*);
	// 下面2行暂时顶替一下
	//virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const;
	//virtual bool bounding_box(double time0, double time1, AABB& output_box) const;
private:
	std::vector<Triangle::Ptr> faces;
};

bool processMesh(const aiMesh* meshPtr, const aiScene* scene)
{
	if (!meshPtr || !scene) return false;
	std::vector<Triangle::Ptr> faces;
	std::vector<Vertex> vertData;
	for (int i = 0; i < meshPtr->mNumVertices; i++)
	{
		Vertex vertex;
		if (meshPtr->HasPositions())
		{
			vertex.position.e[0] = meshPtr->mVertices[i].x;
			vertex.position.e[1] = meshPtr->mVertices[i].y;
			vertex.position.e[2] = meshPtr->mVertices[i].z;
		}
		if (meshPtr->HasTextureCoords(0))
		{
			vertex.tex_coord.e[0] = meshPtr->mTextureCoords[0][i].x;
			vertex.tex_coord.e[1] = meshPtr->mTextureCoords[0][i].y;
		}
		if (meshPtr->HasNormals())
		{
			vertex.normal.e[0] = meshPtr->mNormals[i].x;
			vertex.normal.e[1] = meshPtr->mNormals[i].y;
			vertex.normal.e[2] = meshPtr->mNormals[i].z;
		}
		vertData.push_back(vertex);
	}
	
	auto material = Lambertian::create(color(1, 0, 0));
	for (int i = 0; i < meshPtr->mNumFaces; i++)
	{
		aiFace face = meshPtr->mFaces[i];
		Triangle triangle(vertData[face.mIndices[0]], vertData[face.mIndices[1]], vertData[face.mIndices[2]], material);
		Triangle::Ptr prt(&triangle);
		faces.push_back(prt);
	}
}

bool processNode(const aiNode* node, const aiScene* scene)
{
	if (!node || !scene) return false;
	for (int i = 0; i < node->mNumMeshes; i++)
	{
		const aiMesh* meshPtr = scene->mMeshes[node->mMeshes[i]];
		processMesh(meshPtr, scene);
	}
	for (int i = 0; i < node->mNumChildren; i++)
		processNode(node->mChildren[i], scene);
	return true;
}

void loadModel(std::string path)
{
	if (!fs::is_regular_file(path)) {
		std::cerr << "ERROR::loadModel: No such file " << path << std::endl;
	}
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path,
		aiProcess_Triangulate | aiProcess_FlipUVs);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "ERROR::Assimp: " << importer.GetErrorString() << std::endl;
		return;
	}

	//processNode(scene->mRootNode, scene);
	
	/*else {
		std::cout << "Assimp: OK\n";
		std::cout << "Meshes Number: " << scene->mNumMeshes << std::endl
			<< "Camera Number: " << scene->mNumCameras << std::endl
			<< scene->mMeshes[0]->mNumFaces << std::endl
			<< scene->mMeshes[0]->mNumVertices << std::endl;
		auto uv = scene->mMeshes[0]->mTextureCoords[0][0];
		std::cout << uv.x << ' ' << uv.y << ' ' << uv.z << std::endl;

		for (int j = 0; j < scene->mMeshes[0]->mNumFaces; j++) {
			printf("\n\n====== Face %d ======\n", j);
			auto face = scene->mMeshes[0]->mFaces[j];
			for (int i = 0; i < face.mNumIndices; i++) {
				auto idx = face.mIndices[i];
				auto vertex = scene->mMeshes[0]->mVertices[idx];
				printf("Vertex %d: Idx %d, (%lf, %lf, %lf)\n", i, idx, vertex.x, vertex.y, vertex.z);
				if (scene->mMeshes[0]->HasNormals()) {
					auto norm = scene->mMeshes[0]->mNormals[idx];
					printf("Normals: (%lf, %lf, %lf)\n", norm.x, norm.y, norm.z);
				}

				else {
					printf("No normals\n");
				}
				if (scene->mMeshes[0]->HasTextureCoords(0)) {
					auto uv = scene->mMeshes[0]->mTextureCoords[0][idx];
					printf("UV: (%lf, %lf)\n", uv.x, uv.y);
				}
				else {
					printf("No UV\n");
				}
			}
		}
			
	}*/
}

#endif