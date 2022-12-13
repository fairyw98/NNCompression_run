# NNCompression
Coming soon...
## 基于SMBO的序贯策略优化——贝叶斯优化
贝叶斯优化(Bayesian optimization,简称 BO)是一种有效的解决方法[1].

贝叶斯优化在不同的领域也称作序贯克里金优化(sequential Kriging optimization,简称 SKO)、基于模型的序贯
优化(sequential model-based optimization,简称 SMBO)、高效全局优化(efficient global optimization,简称 EGO).
,贝叶斯优化(Bayesian Optimization)充分利用了上一个点的信息,找到下一个测试的点,能够在很少的评估代价下得
到一个近似最优解.

贝叶斯优化之所以称作“贝叶斯”,是因为优化过程中利用了著名的“贝叶斯定理”

贝叶斯优化主要包含两个核心部分:概率代理模型(Probabilistic Surrogate Model)和采集函数(Acquisition Function,AC)。 
贝叶斯优化本质上更偏向于减少评估的代价,使得优化的过程能够经过较少次数的评估得到最优解。 在每
次迭代的过程中,算法都会对采集函数的最大值的数据点即最有“潜力”的点进行评估,最终收敛到最优解。

概率代理模型是采用一个概率模型来代理目标函数 f(x),而高斯过程则是较为常用的一种模型。 高斯过程
(Gaussian processes, GP)可视为多元高斯概率分布的范化。

由于高斯过程本身有陷入局部最优的问题,因此需要通
过采集函数来寻找下一个最优值,采集函数的目的在于平衡
探索(exploration)和利用( exploitation)两者的选择。 “探索”
目的在于尽量选择远离已知样本点的数据点用作下一次迭
代;“利用”目的在于尽量选择接近已知样本点的数据点用作
下一次迭代。 常用的采集 函 数 有 UCB ( Upper confidence
bound)、PI(Probability of improvement)、EI(Expected improvement)三种

根据以上特点分析,贝叶斯优化适合求解优化目标存在多峰、非凸、黑箱、存在观测噪音并且评估代价高
昂等特点的问题,这些需要我们根据具
体问题选择合适的模型代理模型和采集策略,才能充分发挥贝叶斯优化方法的潜力. 

## 基于TPE方法的贝叶斯优化

在实际使用中，相比基于高斯过程的贝叶斯优化，基于高斯混合模型的TPE在大多数情况下以更高效率获得更优结果，