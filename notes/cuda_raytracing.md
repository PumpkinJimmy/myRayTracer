# CUDA RayTracing

实际的实现方法是把递归展开成循环。

CUDA BVH其实是二叉树非递归前序遍历的变形:
- 手动维护栈,每次从栈顶出栈,检查节点,相交就把左右孩子都入栈即可否则无视