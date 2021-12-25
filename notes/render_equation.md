# 渲染方程

## 简介

渲染方程是形如这样的递归积分方程：
$$
    L_o(p,w_o) = L_e(p,w_o) + \int_{\Omega} L_i(p, w_i) f(p,w_i,w_o) \max(0, \cos(w_i, n)) \mathrm{d} w_i
$$

符号解释：
- $w_o$ 出射方向
- $w_i$ 入射方向
- $p$ 交点
- $n$ 向量
- $L_i$ 从方向$w_i$入射到$p$处的光强
- $L_e$ 物体本身在$p$点向$w_o$发射的光强
- $L_o$ 物体在点$p$处辐射出来的光强
- $f$ 所谓的**BRDF函数**。可以理解为在点$p$处从$w_i$方向入射的光线在$w_o$方向出射强度的“衰减系数”。 

注意到$L_i(p,w_i) = \int_{\Omega} L_o(p, w_i) \mathrm{d}p$，因此该方程还是一个*递归方程*

理论上：**任意复杂的渲染模型都可以使用该方程来表达，且渲染场景的工作可以形式化为在给定参数下对所有点求解渲染方程**

## BxDF

BRDF( Bidirectional Reflectance Distribution Function)


如果考虑折射而不是反射，则是BTDF.

由于完整的宏观的出射主要包括反射和折射，因此将两者同时考虑起来的分布函数称为*BSDF(Bidirectional Scatter Distribution Function)*

**BSDF函数实质上就是我们通常说的“材质”**

举例：
- Lambertian：最理想化的漫反射模型的BSDF就是输入方向无关的均匀分布
- Mirror: 镜面反射的BSDF就是按照反射定理集中在一点的一个脉冲
- Metal: 综合考虑由于粗糙导致的漫反射以及金属表面的镜面反射的BSDF
- Glass: 透明，只考虑折射（当然还有复杂的全反射）

冯氏光照可以认为是渲染方程的特例