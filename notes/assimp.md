# Assimp Notes

## API
```c++
Assimp::Importer importer;
importer.ReadFile(path, aiProcess_Triangulate | aiProcess_FlipUVs);
```

## DataStructure
- `aiScene`
  - mMeshes: Array of `aiMesh:Ptr` & mNumMeshes
- `aiMesh` 
  - mVertices: Array of `aiVector3D`
  - mNormals: Array of `aiVertor3D`
  - mTextureCoords: Array of `array of aiVertor3D`
  - 三个数组的长度都是mNumVertices
  - mFaces: Array of `aiFace`, 数组长度mNumFaces
  - mNormals, mTexureCoords不一定存在，不存在的时候为空指针。
- `aiVector3t<T>`
  - x,y,z
- vertices: spatial positions
- normals: normal vectors
- text coords: 虽然是`aiVertor3D`，但实际上`x,y`分量就是`u,v`分量
- `aiFace`
  - mIndices: Array of `unsigned` as indices in mVertices
  - mNumIndices