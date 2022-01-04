# 三角形相关算法

参考资料：[三角形求交算法](https://blog.csdn.net/zhanxi1992/article/details/109903792)

## 三角形求交：Muller-Trumbore算法
令$\mathbf{O, D, P_0, P_1 P_2}$分别表示射线的原点、射线的方向向量、三角形的三个点。

则当光线与三角形所在平面相交的时候，有如下向量等式成立：
$$
    \mathbf{O} + t\mathbf{D} = (1 - b_1 - b_2) \mathbf{P_0} + b_1 \mathbf{P_1} + b_2 \mathbf{P_2}
$$

解方程得：
$$
    \left[\begin{matrix}t \\ b_1 \\ b_2\end{matrix}\right] 
    = \frac{1}{\mathbf{S_1} \cdot \mathbf{E_1}}
    \left[\begin{matrix}
    \mathbf{S_2} \cdot \mathbf{E_2} \\
    \mathbf{S_1} \cdot \mathbf{S} \\
    \mathbf{S_2} \cdot \mathbf{D}
    \end{matrix}\right]
$$

其中：
$$
\begin{cases}
\mathbf{E_1} = \mathbf{P_1} - \mathbf{P_0}\\
\mathbf{E_2} = \mathbf{P_2} - \mathbf{P_0}\\
\mathbf{S} = \mathbf{O} - \mathbf{P_0}\\
\mathbf{S_1} = \mathbf{D} \times \mathbf{E_2}\\
\mathbf{S_2} = \mathbf{S} \times \mathbf{E_1}
\end{cases}
$$

## 算法数值问题
以下**假设三角形不存在三点共线**
- $\mathbf{S_1}$为0意味着**射线与三角片平行**
- $\mathbf{S_1}$为0且$\mathbf{S_2} \cdot \mathbf{E_2}$为0意味着**射线在三角片的面内**