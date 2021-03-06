基于渲染方程,我们知道了材质的外观最主要取决于BSDF函数的性质.

回忆:
- BSDF = BRDF + BTDF
- **只有透明材质才会涉及BTDF计算**

基于渲染方程,我们提出一个**统一的BSDF函数,可以参数化地描述金属\非金属材质**(Cook-Torrance):
- 给定点的辐射度有漫反射分量和高光反射加权求和组成
- 漫反射采用的是Lambert模型
- 高光反射部分采用一个复杂的,考虑菲涅尔函数,表面粗糙度以及微表面遮蔽的的模型
- 折射部分同样基于菲涅尔
- 菲涅尔部分都利用Schlick的近似公式计算

这就是PBR,基于物理的渲染.

**PBR的BSDF函数**:
$$
    f(w_o, w_i) = f_d(w_o, w_i) + \frac{F(w_0, h) D(h) G(w_i,w_o,h)}{4(n\cdot w_i)(n\cdot w_o)}
$$

其中各项的意义:
- $f_d$的意义是理想Lambert漫反射
- $F$是*菲涅尔项*,表示的是**出射光线中反射所占比例**,
  - 它可以对应生活中一个有趣的现象:视线垂直玻璃时玻璃几乎是透明的;平行玻璃时玻璃几乎像一块镜子.也即,在透明材质中折射和反射的占比取决于观察角度
  - 在实际计算菲涅尔项的时候采用的是Schlick近似函数
- $D$是*法线分布项*,表示的是**h偏离法线的程度**,它与*粗糙程度*密切相关. 依据微表面模型,越粗糙的表面其法线越散乱无章
- $G$表示*遮挡项*,它表现的是**由于微表面相互遮蔽导致的能量耗散**,它主要受粗糙程度影响

但这些还不够,光线追踪还需要采样方向,而不仅仅是数值.这就涉及基于pdf的重要性采样加快收敛.

对于*漫反射和镜面反射同时存在的情况,每次依据比例随机选一种来产生出射方向*

总的来说,基于PBR的光线追踪渲染核心思路是:
1. 对于非透明材质,根据**BRDF函数计算每一步的衰减系数**,然后采用**重要性采样随机采样一个反射方向**
2. 对于透明材质,根据**菲涅尔公式随机选择一个反射或折射方向**,然后令衰减系数为1(完全不衰减)

技巧:
1. 在非透明材质,先采样直接光照明(乘上衰减).因为直接光照是绝大部分的来源,且直接光照是总的光照的一部分,不会让估计有偏（不会算重复的，因为直接光照和最终反射得来的光照性质不同
2. 蒙特卡洛采样只需要两帧的buffer使用交替就可以一直累积下去

## 菲涅尔方程详解
- 菲涅尔现象：对于透明物体表面，垂直入射的时候大部分光线被折射呈“透明”效果；平行入射的时候大部分光线被反射呈“镜子”效果
- 菲涅尔方程：导出一个函数，输入界面两侧物体的折射率和视线与法向量的夹角（入射角），输出被反射的光线所占的比例
- 基础反射率：入射角为90度的时候的反射比例。
- Schlick近似：对原始菲涅尔函数做近似，输入基础反射率和入射角，输出近似的菲尼尔函数值
- 基于折射率之比可以直接计算基础反射率（非金属）。
- 问题：**Schlick只对非金属有效**。金属材料的折射率是负数，因此无法直接计算基础反射率。
- 实际上，**不透明表明上的折射光变成了漫反射。**。不透明材质直接使用菲涅尔公式计算出漫反射-镜面反射比例

- **金属由于吸收了折射光而导致几乎没有漫反射**
- 由此，我们预先根据材质选定一个基础反射率，然后根据参数*金属度*对基础反射率-表面色彩做线性插值。当金属度为1时，F0就是表面色彩；当金属度为0时，F0是基础反射率
- 综上，有*金属/非金属不透明材质统一Schlick近似*