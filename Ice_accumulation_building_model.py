#假霜点温度法,SFIP指数，积冰速率法
import numpy as np
import pandas as pd
from T_d import calculate_dew_point_from_water_content 
from T_d import calculate_relative_humidity
def Tfi(V,T,LWC):
#其中Tfi为假霜点温度，V为飞机空速（单位：100km/h）
#T为温度（单位：℃），Td为露点温度（单位：℃）
    V = V * 3600 / 100000 #单位换算 
    Td = calculate_dew_point_from_water_content(LWC,T)
    Td = Td + 273.15
    fig = ''
    T_fi = -0.15*(V/100)**2*(T-Td)
    #T_fi为摄氏度,前面T为K单位,需要换算成摄氏度
    T = T - 273.15
    if T_fi - T <= -0.15 * (V/100)**2:
        fig = '无积冰'
    else:
        fig = '有积冰'
    if T_fi - T > 0:
        fig = '中度即以上积冰'
    return fig,T_fi
def SFIP(T,w,LWC):
    RH = calculate_relative_humidity(LWC,T)/100
    T= T - 273.15
    a = 0.35
    b = 0.2
    c = 0.45
    #计算MT温度系数
    MT = 0  #先声明MT变量
    T1,T2,T3,T4= -28,-12,-1,1 #单位：℃
    if T <= T1:
        MT = 0
    elif T1 < T <= T2:
        MT = (T-T1)/(T2-T1)
    elif T2 < T <= T3:
        MT = 1;
    elif T3 < T <= T4:
        MT = 1-(T-T3)/(T4-T3)
    elif T>T4:
        MT = 0
    ########################
    #计算MRH湿度系数
    MRH = 0 #先声明MRH变量
    RH1,RH2 = 0.6,0.95  #RH为湿度
    if RH <= RH1:
        MRH = 0
    elif RH1 < RH <= RH2:
        MRH = ((RH-RH1)/(RH2-RH1))**2
    elif RH > RH2:
        MRH = 1
    #########################
    #计算MW垂直速度系数
    MW = 0 #先声明MW变量
    w1,w2,w3 = -0.1,0,0.5 #w为垂直速度（单位：m/s）
    if w <= w1:
        MW = -0.4
    elif w1 < w <= w2:
        MW = -0.4 + 0.4 * (w-w1)/(w2-w1)
    elif w > w3:
        MW = 1
    #########################
    #计算MLWC液态水含量系数
    MLWC = 0 #先声明MLWC变量
    LWC0 = 0.4 # LWC为液态水含量（单位：g/kg）
    if LWC <= LWC0:
        MLWC = LWC/0.4
    elif LWC > LWC0:
        MLWC = 1
    #########################
    fig = ''
    sfip = MT*(a*MRH+b*MW+c*MLWC)
    if sfip < 0:
        sfip = 0
    if 0 <= sfip < 0.4:
        fig = '无积冰'
    elif 0.4 <= sfip < 0.6:
        fig = '轻度积冰'
    elif 0.6 <= sfip < 0.8:
        fig = '中度积冰'
    elif 0.8 <= sfip <=1:
        fig = '严重积冰'
    return fig,sfip
def Rice(LWC,V,C): #Rice为积冰速率（单位：cm/hr）
#A＝3.6×105为单位转化常数，LWC为液态水含量（单位：g/m3）
#V为飞机空速（单位：m/s），β为过冷水滴收集系数，
#取β=0.80088，ρ为累积冰层的密度（单位：g/m3），取ρ=9.17×105g/m3
    A = 3.6*10**5
    β = 0.58
    ρ = 9.17*10**5
    fig = ''
    if C == 0.3048:
        R_ice = A*LWC*V*β/ρ
    else:
        R_ice = 0.675*A*LWC*V*β/ρ
    if R_ice < 0.6:
        fig = '无积冰'
    elif 0.6 <= R_ice < 2.5:
        fig = '轻度积冰'
    elif 2.5 <= R_ice < 7.5:
        fig = '中度积冰'
    elif R_ice >= 7.5:
        fig = '重度积冰'
    return fig,R_ice 
# 读取数据文件
df = pd.read_csv('combine_1.csv',encoding = 'GB2312')
'''
# 执行计算并添加新列
df['Tfi'], df['Tfi_fig'] = zip(*df.apply(
    lambda row: Tfi(row['V'], row['T'], row['W']), axis=1))

df['SFIP'], df['SFIP_fig'] = zip(*df.apply(
    lambda row: SFIP(row['T'], 0 , row['W']), axis=1))
'''
df['Rice'], df['Rice_fig'] = zip(*df.apply(
    lambda row: Rice(row['W'],row['V'],row['C']), axis=1))

# 保存结果到新文件（避免覆盖原始数据）
df.to_csv('Rice.csv', index=False)