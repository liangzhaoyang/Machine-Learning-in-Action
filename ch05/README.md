# Logistic �ع�
����˵������ʵ���ǡ��߼��ع顱��

## ԭ��
����ʹ�� Logistic �ع������������⡣���������ֱ�Ϊ 0 �� 1��Logistic �ع���Ϊ����������� 1 �ĸ��� ![\\inline P\(y=1|x,\\theta\) = \\text{sigmoid}\(x\\theta\)](http://latex.codecogs.com/png.latex?%5Cinline%20P%28y%3D1|x%2C%5Ctheta%29%20%3D%20%5Ctext{sigmoid}%28x%5Ctheta%29)������ ![\\inline \\text{sigmoid}\(x\) = \\frac{1}{1+e^{-x}}](http://latex.codecogs.com/png.latex?%5Cinline%20%5Ctext{sigmoid}%28x%29%20%3D%20%5Cfrac{1}{1%2Be^{-x}})��![\\inline x=\(x_0,x_1,x_2,\\dots,x_m\)](http://latex.codecogs.com/png.latex?%5Cinline%20x%3D%28x_0%2Cx_1%2Cx_2%2C%5Cdots%2Cx_m%29) ��һ������������ʾһ�������ĸ���������ֵ��Ϊ�˷���������賣����![\\inline x_0=1](http://latex.codecogs.com/png.latex?%5Cinline%20x_0%3D1)��![\\inline \\theta](http://latex.codecogs.com/png.latex?%5Cinline%20%5Ctheta) ��һ������������ʾÿ�������Խ���ġ�Ȩ�ء���

�蹲�� ![n](http://latex.codecogs.com/png.latex?n) ��ѵ��������ÿ�������� ![m](http://latex.codecogs.com/png.latex?m) ���������� ![\\inline P\(y=1|x,\\theta\) = h](http://latex.codecogs.com/png.latex?%5Cinline%20P%28y%3D1|x%2C%5Ctheta%29%20%3D%20h)��Logistic �ع���Ϊ����ѵ�������Ĵ��ۺ���Ϊ

![L = -\\frac{1}{n}\\sum_{i=1}^n\(y_i\\text{log}\(h\)+\(1-y_i\)\\text{log}\(1-h\)\)](http://latex.codecogs.com/png.latex?L%20%3D%20-%5Cfrac{1}{n}%5Csum_{i%3D1}^n%28y_i%5Ctext{log}%28h%29%2B%281-y_i%29%5Ctext{log}%281-h%29%29)

��������Իع�һ��ȡƽ�����������ۺ������ᵼ�´��ۺ�����͹�������Ż���

ʹ���ݶ��½����Ż����ۺ������� ![h](http://latex.codecogs.com/png.latex?h) չ�������ԶԴ��ۺ����е�ĳһ������ ![x_j](http://latex.codecogs.com/png.latex?x_j) �󵼣��ᷢ�ַǳ�Ư���Ľ����

![\(y_i\\text{log}\(h\)\)' = y_i\(1-h\)x_j](http://latex.codecogs.com/png.latex?%28y_i%5Ctext{log}%28h%29%29%27%20%3D%20y_i%281-h%29x_j)

![\(\(1-y_i\)\\text{log}\(1-h\)\)' = \(y_i-1\)hx_j](http://latex.codecogs.com/png.latex?%28%281-y_i%29%5Ctext{log}%281-h%29%29%27%20%3D%20%28y_i-1%29hx_j) 

��Ӽ��ɵõ����ۺ������ݶ�

![L' = -\\frac{1}{n}\\sum_{i=1}^n\\sum_{j=1}^m\(y_i-h\)x_j](http://latex.codecogs.com/png.latex?L%27%20%3D%20-%5Cfrac{1}{n}%5Csum_{i%3D1}^n%5Csum_{j%3D1}^m%28y_i-h%29x_j)

��һ���ǳ�Ư�����ݶȣ����������ݶ��½�������ݶ��½����ɡ�

## Ӧ��

������������ѵ�����ϣ��������������ǳ�����

![gradient descent result](screenshot/gradientDescent.png)

^ ��ͨ�ݶ��½����

![random gradient descent result](screenshot/randGradientDescent.png)

^ ����ݶ��½����

�����ڲ�������ݼ���׼ȷ��ֻ�� 77% ���£������е�˵�������ڲ������ݵĲ�������ȱʧ��������ֵ����ȥ�Ľ��...
