# IYPT 2019 循环摆计算程序实例

## 题目回顾

> **Problem No.14 “Looping pendulum”**
>
> Connect two loads, one heavy and one light, with a string over a horizontal rod 
> and lift up the heavy load by pulling down the light one. Release the light load 
> and it will sweep around the rod, keeping the heavy load from falling to the ground.
> Investigate this phenomenon.

## 文件说明

- `looping_pendulum.py` 一个包含循环摆系统的程序源码。
- `env.yml` 一个适用于本程序的`conda`虚拟环境。
- `循环摆的Lagrangian方法.pdf` 求解程序说明。

## 依赖

- `python`, >3.0
- `numpy`
- `scipy`, 需要`scipy.integrate.ode`
- `matplotlib`, 用于绘图
- `pandas`, 用于导出求解得到的数据(`.csv`文件)
