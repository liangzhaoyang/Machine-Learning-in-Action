# Ch04 - 朴素贝叶斯分类器

## 原理
朴素贝叶斯分类器通过求出使得概率 ![P\(X|W\)](http://latex.codecogs.com/gif.latex?P%28X|W%29) 最大化的类别 ![X](http://latex.codecogs.com/gif.latex?X)，以确定特征向量 ![W = \(w_1, w_2, w_3, \\dots\)](http://latex.codecogs.com/gif.latex?W%20%3D%20%28w_1%2C%20w_2%2C%20w_3%2C%20%5Cdots%29) 最有可能属于的类别。

根据条件概率公式，![P\(X|W\) = \\frac{P\(W|X\)P\(X\)}{P\(W\)}](http://latex.codecogs.com/gif.latex?P%28X|W%29%20%3D%20%5Cfrac{P%28W|X%29P%28X%29}{P%28W%29})。![P\(X\)](http://latex.codecogs.com/gif.latex?P%28X%29)可以视为一个先验概率，用类别 ![X](http://latex.codecogs.com/gif.latex?X) 在样本中的频率近似算出。![P\(W\)](http://latex.codecogs.com/gif.latex?P%28W%29) 虽然很难计算，但它是一个与 ![X](http://latex.codecogs.com/gif.latex?X) 无关的常数，而我们只需要找到使得概率最大化的 ![X](http://latex.codecogs.com/gif.latex?X)，只要比较大小，并不需要精确算出这个概率，所以可以无视这个值。

问题就在于如何计算 ![P\(W|X\)](http://latex.codecogs.com/gif.latex?P%28W|X%29)，这里就是朴素贝叶斯分类器的“朴素”体现出来的地方。朴素贝叶斯分类器做了一个强假设，认为 ![W](http://latex.codecogs.com/gif.latex?W) 里的每个特征都是互相独立的，即 ![P\(W|X\) = P\(w_1|X\)P\(w_2|X\)P\(w_3|X\)\\dots](http://latex.codecogs.com/gif.latex?P%28W|X%29%20%3D%20P%28w_1|X%29P%28w_2|X%29P%28w_3|X%29%5Cdots)，这就方便了我们的概率计算。

为了计算某一个特征的概率 ![P\(w|X\)](http://latex.codecogs.com/gif.latex?P%28w|X%29)，如果 ![w](http://latex.codecogs.com/gif.latex?w) 的取值是离散的，直接使用古典概型计算即可；如果 ![w](http://latex.codecogs.com/gif.latex?w) 的取值是连续的，可以假设 ![w](http://latex.codecogs.com/gif.latex?w) 服从正态分布。

太多的概率乘起来，可能会因为结果太小导致下溢。可以将概率取对数，这样乘法就变成了加法。

## 应用：判别是否为垃圾邮件

统计出样本中出现过哪些词，我们就能将每封邮件转化为一个词向量。用这个词向量放进朴素贝叶斯分类器里处理即可。

我遇到的问题是计算某个词在某一类别里出现的概率。虽然某个词出现的次数是离散量，但是这个离散量有无数种可能的取值，所以我还是使用正态分布来计算概率。

但是正态分布不能计算“恰好等于某个值”的概率，因为正态分布表现的是连续分布的随机变量的概率密度。所以我将“某个词出现 ![n](http://latex.codecogs.com/gif.latex?n) 次的概率”转化为“某个词出现次数在 ![\[n-0.5, n+0.5\)](http://latex.codecogs.com/gif.latex?%5Bn-0.5%2C%20n%2B0.5%29)之间的概率”，这样就能用正态分布计算了。

不过正态分布的积分很困难，我只能以高斯函数在 ![x = n](http://latex.codecogs.com/gif.latex?x%20%3D%20n) 处的取值 ![y](http://latex.codecogs.com/gif.latex?y) 作为近似的概率（因为区间宽度是 1，实际上是一个宽度为 1，高度为 ![y](http://latex.codecogs.com/gif.latex?y) 的矩形的面积）。不过测试起来准确度还不错，不知道是本来就可以这样近似，还是书的作者选数据时特意选择了增强读者信心的数据...
