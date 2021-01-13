# coding: utf-8
#
# looping_pendulum.py
# 作者 : zhiyuanzhai
# 创建日期 : 2019.5.
#
# ********************** 题目回顾 **********************
#
# 14. Looping Pendulum
# Connect two loads, one heavy and one light, with a string over a horizontal rod 
# and lift up the heavy load by pulling down the light one. Release the light load
# and it will sweep around the rod, keeping the heavy load from falling to the ground. 
# Investigate this phenomenon.
#
# 14. 循环摆
# 将一重一轻两个负载通过水平杆的一根绳子相连，并下拉轻负载以吊起重，释放轻负载，它会围着杆扫动，从而阻止
# 重的负载下降到地面。探究这种现象。
# 
# ********************** 程序内容 **********************
# 
# 1. 一个描述运动状态的「类」: motion
# 2. 循环摆物件的类，2个，对应两个运动阶段
# 3. 循环摆运动的求解器，包含两个运动阶段
# 4. 测试求解结果和输出功能的代码，该部分不可被其他文件调用
# 
# ********************** 采用单位 **********************
# 
# 程序采用 cm, s, g 为主单位。
# 

from math import cos, sin, exp
from time import localtime, strftime

import matplotlib.pyplot as plt
import numba as nb
import numpy as np
import pandas as pd

from math import pi
from scipy.integrate import ode
from scipy.linalg import solve

# 定义普适常数及相关参数。
g = 980.0               # 重力加速度。

class motion :
    
    def __init__(self, T, U, hold, R) :
        self.time, self.motion, self.hold, self.R = T, U, hold, R
        self.ending = T[-1]
    
    def light_load_path(self) :

        L = self.motion[:,0]
        Phi = self.motion[:,1]
        X = self.R*np.cos(Phi) - L*np.sin(Phi)
        Y = self.R*np.sin(Phi) + L*np.cos(Phi)

        return pd.DataFrame(
            np.vstack([self.time, X, Y]).transpose(),
            columns = ["t/s", "X/cm", "Y/cm"]
        )
    
    def plot_light_load_path(self, Axes) :

        path = self.light_load_path()

        if self.hold :
            line_color = "yellowgreen"
        else :
            line_color = "skyblue"
        Axes.plot(path["X/cm"], path["Y/cm"], color=line_color)

        return Axes

    def heavy_load_drop(self, S=0.0) :

        H = S - (self.motion[:,0] + self.R * self.motion[:,1])

        return pd.DataFrame(
            np.vstack([self.time, H]).transpose(), 
            columns = ["t/s", "H/cm"]
        )

    def plot_heavy_load_drop(self, Axes, S=0.0, plot_static_state=True) :

        drop = self.heavy_load_drop(S)

        if self.hold :
        	if not plot_static_state :
        		return Axes
        	line_color = "yellowgreen"
        else :
            line_color = "skyblue"
        
        Axes.plot(drop["t/s"], drop["H/cm"], color=line_color)
        
        return Axes

    def original_data_output(self) :
        return pd.DataFrame(
            np.vstack([self.time, self.motion.transpose()]).transpose(), 
            columns = ["t/s", "l(cm)","phi(rad)","vl(cm/s)","vphi(rad/s)"]
        )


# 定义循环摆「物件」类。

class Looping_Pendulum_Stage_1 :

    def __init__(self, parameters, direction) :
        self.m, self.M, self.u, self.R, self.c1, self.c2 = parameters
        self.direction = direction
    
    def f(self, t, x) :
        l, phi, vl, vphi = x.tolist()

        Dl = vl
        Dphi = vphi
        
        M = self.M * exp(self.direction*self.u*phi)
        m = self.m
        
        A = np.mat([[    m + M          ,   (m + M)*self.R              ],
                    [   (m + M)*self.R  ,   m*l**2+(m + M)*self.R**2    ]])
        I = np.mat([[    m*l*vphi**2    ],
                    [  - 2*m*l*vl*vphi  ]])
        G = np.mat([[  - M*g - m*g*cos(phi)                              ],
                    [  - m*g*(self.R*cos(phi)-l*sin(phi)) - M*g*self.R   ]])
        D = np.mat([[  - (self.c1 + self.c2)*(vl + self.R*vphi)                             ],
                    [  - self.R*(self.c1 + self.c2)*(vl + self.R*vphi) - self.c1*(l**2)*vphi]])
        x = solve(A,I+G+D)
        
        Dvl, Dvphi = x[0], x[1]

        return [Dl, Dphi, Dvl, Dvphi]


class Looping_Pendulum_Stage_2 :

    def __init__(self, parameters) :
        self.l0, self.phi0, self.R, self.c1 = tuple(parameters)
    
    def f(self, t, x) :
        Phi, v = x.tolist()
        l = self.l0 - self.R * (Phi-self.phi0)
        DPhi = v
        Dv = (g*sin(Phi) + self.R*v**2 - self.c1*(l**2)*v) / l
        return [DPhi, Dv]


def Solve_Stage_1(initials, parameters, direction, t0, T=5.0) :
    """
    求解第一阶段轻负载运动. 所需参数：
    * initials - 初始条件, list. 需要包含4个元素，依次是：
        轻负载端初始绳长, 绳在杆上缠绕的初始角度, 轻负载沿绳方向的初速度, 绳在杆上缠绕角度变化的角速度.
    * parameters - 参数. 需要包含6个元素, 依次是: 
        轻负载质量, 重负载质量, 绳与杆之间的摩擦系数, 杆半径, 轻负载的阻尼系数, 重负载的阻尼系数。
    """
    initials = list(initials)
    R = parameters[3]
    timescale = (R/g)**0.5

    system = Looping_Pendulum_Stage_1(parameters, direction)
    r = ode(system.f)
    r.set_initial_value(initials, t0)
    r.set_integrator("vode", method="bdf")
    dt = 1.0e-2 * timescale

    t = [t0,]
    U = [initials,]
    if direction == - 1 :
        while r.successful() :
            if r.t+dt >= T or r.y[0] < 0 or r.y[2] > 0 :
                hold = False
                stop = True
                break
            if (r.y[2] + R*r.y[3]) > 0 :
                hold = True
                stop = False
                break
            r.integrate(r.t+dt)
            t.append(r.t)
            U.append(r.y)
    elif direction == 1 :
        while r.successful() :
            if r.t+dt >= T or r.y[0] < 0 or r.y[2] > 0 :
                hold = False
                stop = True
                break
            if (r.y[2] + R*r.y[3]) < 0 :
                hold = True
                stop = False
                break
            r.integrate(r.t+dt)
            t.append(r.t)
            U.append(r.y)
    
    return np.array(U), np.array(t), hold, stop


def Solve_Stage_2(initials, parameters, t0, T=5.0) :
    """
    求解第二阶段轻负载运动. 所需参数:
    * initials - 初始条件, list. 需要包含4个元素, 依次是：
        轻负载端初始绳长, 绳在杆上缠绕的初始角度, 轻负载沿绳方向的初速度, 绳在杆上缠绕角度变化的角速度
    * parameters - 参数. 需要包含5个元素，依次是: 
        轻负载质量, 重负载质量, 绳与杆之间的摩擦系数, 杆半径, 轻负载的阻尼系数
    """
    l0, phi0 = initials[0], initials[1]
    initials = [initials[1], initials[3]]
    m, M, u, R, c1 = tuple(parameters)
    parameters = (l0, phi0, R, c1)
    direction = -1
    timescale = (R/g)**0.5

    system = Looping_Pendulum_Stage_2(parameters)
    q = ode(system.f)
    q.set_initial_value(initials, t0)
    q.set_integrator("vode", method="bdf")
    dt = 1.0e-2 * timescale

    t = [t0,]
    U = [initials,] 
    while q.successful() :
        if q.t+dt >= T or (l0 - R*(q.y[0]-phi0)) < 0 or q.y[1] > 500 :
            stop = True
            hold = True
            break
        F = - m*g*cos(q.y[0]) + m*(l0 - R*(q.y[0]-phi0))*q.y[1]**2
        if F < M*g*exp(-u*q.y[0]) :
            stop = False
            hold = False
            direction = -1
            break
        if F > M*g*exp(u*q.y[0]) :
            stop = False
            hold = False
            direction = 1
        q.integrate(q.t+dt)
        t.append(q.t)
        U.append(q.y)

    U = np.array(U)
    U = np.vstack((l0-R*(U[:,0]-phi0), U[:,0], -R*U[:,1], U[:,1])).transpose()

    return U, np.array(t), hold, stop, direction


def Main_Process(initials, parameters, hold=False, t0=0.) :
    """
    控制运动两个阶段相互转换的过程。
    输入参数有4个：
    * initials - 运动的初始状态，需要包含4个元素：
        轻负载端初始绳长，绳在杆上缠绕的初始角度，轻负载沿绳方向的初速度，绳在杆上缠绕角度变化的角速度。
    * parameters - 参数，需要包含6个元素：
        轻负载质量，重负载质量，绳与杆之间的摩擦系数，杆半径，轻负载的阻尼系数，重负载的阻尼系数。
    * t0 - 运动的初始时刻，默认为 0.0.
    * hold - 运动初态是否处于第二阶段。

    返回值：
    U - list，其元素是求解所得的 motion 类。
    """

    m, M, u, R, c1, c2 = parameters
    stop = False

    U = []
    direction = -1
    while not stop :
        state = hold
        if hold :
            Us, ts, hold, stop, direction = Solve_Stage_2(initials, (m, M, u, R, c1), t0)
        else :
            Us, ts, hold, stop = Solve_Stage_1(initials, (m, M, u, R, c1, c2), direction, t0)
        initials = Us[-1,:]
        t0 = ts[-1]
        U.append(motion(T=ts, U=Us, hold=state, R=R))

    return U


# 以下是测试程序，以检验的程序功能。

if __name__ == "__main__":

    # 输入参数。
    parameters = (50, 200, 0.1, 1, 0.01, 0.01)
    m, M, u, R, c1, c2 = parameters
    initials = [100, pi/2, 0, 0]
    
    hold = False
    U = Main_Process(initials, parameters, hold)
    
    # 画轻负载轨迹图。
    print("[Matplotlib]正在作轻负载轨迹图...")
    fig, ax = plt.subplots()
    for Motion in U :
        ax = Motion.plot_light_load_path(ax)
    ax.set_xlabel("$X$/cm")
    ax.set_ylabel("$Y$/cm")
    ax.axis("equal")
    ax.legend(["Stage 1", "Stage 2"])
    fig.savefig("test/light.png", dpi=300)

    # 画重负载下落运动的位置-时间图像。
    print("[Matplotlib]正在作重负载下落的位置-时间图...")
    fig, ax = plt.subplots()
    for Motion in U :
        ax = Motion.plot_heavy_load_drop(ax)
    ax.set_xlabel("$t$/s")
    ax.set_ylabel("$H$/cm")
    ax.invert_yaxis()                # 反转y轴。
    fig.savefig("test/heavy.png", dpi=300)

    if U[-1].hold :
        result="重物最终被拉住。"
    else :
        result="重物最终未被拉住。"
    
    date = strftime("%Y.%m.%d.", localtime())
    report = "\n" + date + " " + \
        "运动模拟的输入信息 :" + \
        "\n输入参数\n" + \
        "轻负载质量      :    %.3f\n" % m  + \
        "重负载质量      :    %.3f\n" % M  + \
        "绳杆动摩擦因数  :    %.3f\n" % u  + \
        "杆半径          :    %.3f\n" % R  + \
        "轻负载阻尼      :    %.3f\n" % c1 + \
        "重负载阻尼      :    %.3f\n" % c2 + \
        "\n初始条件:\n" + \
        "%.2f, %.2f, %.2f, %.2f\n" % tuple(initials) + \
        "\n" + result
    print(report)
