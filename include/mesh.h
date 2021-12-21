#pragma once
#ifndef __MESH_H
#define __MESH_H
#include <string>
#include <iostream>
#include <assimp/Importer.hpp>
#include <assimp/scene.h>
#include <assimp/postprocess.h>
#include <filesystem>

namespace fs = std::filesystem;

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
	
	else {
		std::cout << "Assimp: OK\n";
		//std::cout << "Meshes Number: " << scene->mNumMeshes << std::endl
		//	<< "Camera Number: " << scene->mNumCameras << std::endl
		//	<< scene->mMeshes[0]->mNumFaces << std::endl
		//	<< scene->mMeshes[0]->mNumVertices << std::endl;
		//auto uv = scene->mMeshes[0]->mTextureCoords[0][0];
		//std::cout << uv.x << ' ' << uv.y << ' ' << uv.z << std::endl;

		//for (int j = 0; j < scene->mMeshes[0]->mNumFaces; j++) {
		//	printf("\n\n====== Face %d ======\n", j);
		//	auto face = scene->mMeshes[0]->mFaces[j];
		//	for (int i = 0; i < face.mNumIndices; i++) {
		//		auto idx = face.mIndices[i];
		//		auto vertex = scene->mMeshes[0]->mVertices[idx];
		//		printf("Vertex %d: Idx %d, (%lf, %lf, %lf)\n", i, idx, vertex.x, vertex.y, vertex.z);
		//		if (scene->mMeshes[0]->HasNormals()) {
		//			auto norm = scene->mMeshes[0]->mNormals[idx];
		//			printf("Normals: (%lf, %lf, %lf)\n", norm.x, norm.y, norm.z);
		//		}

		//		else {
		//			printf("No normals\n");
		//		}
		//		if (scene->mMeshes[0]->HasTextureCoords(0)) {
		//			auto uv = scene->mMeshes[0]->mTextureCoords[0][idx];
		//			printf("UV: (%lf, %lf)\n", uv.x, uv.y);
		//		}
		//		else {
		//			printf("No UV\n");
		//		}
		//	}
		//}
			
	}
}

#endif