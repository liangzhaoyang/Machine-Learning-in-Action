# Logistic 回归
（据说中文其实不是“逻辑回归”）

## 原理
书中使用 Logistic 回归解决二分类问题。设两种类别分别为 0 和 1，Logistic 回归认为样本属于类别 1 的概率 ![\\inline P\(y=1|x,\\theta\) = \\text{sigmoid}\(x\\theta\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28y%3D1|x%2C%5Ctheta%29%20%3D%20%5Ctext{sigmoid}%28x%5Ctheta%29)。其中 ![\\inline \\text{sigmoid}\(x\) = \\frac{1}{1+e^{-x}}](http://latex.codecogs.com/png.latex?%5Cinline%20%5Ctext{sigmoid}%28x%29%20%3D%20%5Cfrac{1}{1%2Be^{-x}})，![\\inline x=\(x_0,x_1,x_2,\\dots,x_m\)](http://latex.codecogs.com/png.latex?%5Cinline%20x%3D%28x_0%2Cx_1%2Cx_2%2C%5Cdots%2Cx_m%29) 是一个行向量，表示一个样本的各种特征的值。为了方便起见，设常数项![\\inline x_0=1](http://latex.codecogs.com/png.latex?%5Cinline%20x_0%3D1)。![\\inline \\theta](http://latex.codecogs.com/png.latex?%5Cinline%20%5Ctheta) 是一个列向量，表示每种特征对结果的“权重”。

设共有 ![n](http://latex.codecogs.com/png.latex?n) 个训练样本，每个样本有 ![m](http://latex.codecogs.com/png.latex?m) 种特征。设 ![\\inline P\(y=1|x,\\theta\) = h](http://latex.codecogs.com/png.latex?%5Cinline%20P%28y%3D1|x%2C%5Ctheta%29%20%3D%20h)，Logistic 回归认为所有训练样本的代价函数为

![L = -\\frac{1}{n}\\sum_{i=1}^n\(y_i\\text{log}\(h\)+\(1-y_i\)\\text{log}\(1-h\)\)](http://latex.codecogs.com/png.latex?L%20%3D%20-%5Cfrac{1}{n}%5Csum_{i%3D1}^n%28y_i%5Ctext{log}%28h%29%2B%281-y_i%29%5Ctext{log}%281-h%29%29)

如果和线性回归一样取平方函数做代价函数，会导致代价函数非凸，不好优化。

使用梯度下降法优化代价函数。把 ![h](http://latex.codecogs.com/png.latex?h) 展开，尝试对代价函数中的某一个变量 ![x_j](http://latex.codecogs.com/png.latex?x_j) 求导，会发现非常漂亮的结果。

![\(y_i\\text{log}\(h\)\)' = y_i\(1-h\)x_j](http://latex.codecogs.com/png.latex?%28y_i%5Ctext{log}%28h%29%29%27%20%3D%20y_i%281-h%29x_j)

![\(\(1-y_i\)\\text{log}\(1-h\)\)' = \(y_i-1\)hx_j](http://latex.codecogs.com/png.latex?%28%281-y_i%29%5Ctext{log}%281-h%29%29%27%20%3D%20%28y_i-1%29hx_j) 

相加即可得到代价函数的梯度

![L' = -\\frac{1}{n}\\sum_{i=1}^n\\sum_{j=1}^m\(y_i-h\)x_j](http://latex.codecogs.com/png.latex?L%27%20%3D%20-%5Cfrac{1}{n}%5Csum_{i%3D1}^n%5Csum_{j%3D1}^m%28y_i-h%29x_j)

是一个非常漂亮的梯度，用它来做梯度下降和随机梯度下降即可。

## 应用

在两个特征的训练集上，分类结果看起来非常不错。

![gradient descent result](screenshot/gradientDescent.png)

^ 普通梯度下降结果

![random gradient descent result](screenshot/randGradientDescent.png)

^ 随机梯度下降结果

不过在病马的数据集上准确率只有 77% 上下，按书中的说法是由于部分数据的部分特征缺失，用特殊值补进去的结果...
