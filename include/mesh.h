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
#include "bvh.h"

namespace fs = std::filesystem;

class Mesh : public Hittable {
public:

	typedef shared_ptr<Mesh> Ptr;
	Mesh(aiMesh* ai_mesh = nullptr) {
		if (ai_mesh) loadMesh(ai_mesh);
	}
	bool loadMesh(aiMesh* ai_mesh) {
		if (!ai_mesh) return false;
		HittableList hit_faces;
		vec3 min_p(inf, inf, inf), max_p(-inf, -inf, -inf);
		for (int i = 0; i < ai_mesh->mNumVertices; i++)
		{
			auto vertex = make_shared<Vertex>();
			if (ai_mesh->HasPositions())
			{
				vertex->position.e[0] = ai_mesh->mVertices[i].x;
				vertex->position.e[1] = ai_mesh->mVertices[i].y;
				vertex->position.e[2] = ai_mesh->mVertices[i].z;
				min_p = min(min_p, vertex->position);
				max_p = max(max_p, vertex->position);
			}
			if (ai_mesh->HasTextureCoords(0))
			{
				vertex->tex_coord.e[0] = ai_mesh->mTextureCoords[0][i].x;
				vertex->tex_coord.e[1] = ai_mesh->mTextureCoords[0][i].y;
			}
			if (ai_mesh->HasNormals())
			{
				vertex->normal.e[0] = ai_mesh->mNormals[i].x;
				vertex->normal.e[1] = ai_mesh->mNormals[i].y;
				vertex->normal.e[2] = ai_mesh->mNormals[i].z;
			}
			// std::cout << vertex->position << std::endl;
			vertices.push_back(vertex);
		}

		auto material = Lambertian::create(color(0.5, 0, 0.5));
		for (int i = 0; i < ai_mesh->mNumFaces; i++)
		{
			auto face = ai_mesh->mFaces[i];
			Triangle::Ptr prt = make_shared<Triangle>(vertices[face.mIndices[0]], vertices[face.mIndices[1]], vertices[face.mIndices[2]], material);
			// ensure no three point on one line
			auto v0 = vertices[face.mIndices[0]]->position, v1 = vertices[face.mIndices[1]]->position, v2 = vertices[face.mIndices[2]]->position;
			if (cross(v2 - v0, v1 - v0).length_squared() < 1e-15) {
				continue;
			}
			faces.push_back(prt);
			hit_faces.add(prt);
		}
		hit_node = BVHNode(hit_faces, 0, 0);
		bbox = AABB(min_p - vec3(0.0001, 0.0001, 0.0001), max_p + vec3(0.0001, 0.0001, 0.0001));
	}
	virtual bool hit(const ray& r, double t_min, double t_max, hit_record& rec) const {
		return hit_node.hit(r, t_min, t_max, rec);
	}
	virtual bool bounding_box(double time0, double time1, AABB& output_box) const {
		output_box = bbox;
		return true;
	}
	void setMaterial(Material::Ptr mat) {
		for (auto tri : faces) {
			tri->setMaterial(mat);
		}
	}
private:
	std::vector<Vertex::Ptr> vertices;
	std::vector<Triangle::Ptr> faces;
	AABB bbox;
	BVHNode hit_node;
};

Mesh::Ptr loadModel(std::string path)
{
	if (!fs::is_regular_file(path)) {
		std::cerr << "ERROR::loadModel: No such file " << path << std::endl;
	}
	Assimp::Importer importer;
	const aiScene* scene = importer.ReadFile(path,
		aiProcess_Triangulate);
//		aiProcess_Triangulate | aiProcess_FlipUVs);
	if (!scene || scene->mFlags & AI_SCENE_FLAGS_INCOMPLETE || !scene->mRootNode) {
		std::cerr << "ERROR::Assimp: " << importer.GetErrorString() << std::endl;
		return make_shared<Mesh>();
	}
	else {
		std::cout << path << " OK " << std::endl
			<< scene->mMeshes[0]->mNumVertices << " vertices\n"
			<< scene->mMeshes[0]->mNumFaces << " faces\n\n";
	}

	auto mesh = make_shared<Mesh>(scene->mMeshes[0]);
	return mesh;

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