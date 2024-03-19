import gurobipy as gp
from gurobipy import GRB
import math
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import random 
from typing import Union
from collections import defaultdict


def order_delivery_ip(jobs : list[int], orders:dict[str,list[set , int]], processors:dict[str, list[float]], image_name:str) -> None:
    jobs =  {'j'+str(j+1) : jobs[j] for j in range(len(jobs))}
    orders, job_in_order, w = gp.multidict(orders)
    processors, u, f, b, s = gp.multidict(processors)

    # job process time in each processor
    t = {(j,p) : math.ceil(jl/ss) for j, jl in jobs.items() for p, ss in s.items()}

    print('job process time :', t)

    m = gp.Model("parallel_order")
    x = {}
    y = {}
    xo = {}
    L = {}
    C = {}
    z = {}
    phi = {}    
    M = 100000
    
    # decision variables
    for j in jobs:
        for i in processors:
            x[j,i] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{i}")
            for g in orders:
                for gg in orders:
                    z[j,i,g,gg] = m.addVar(vtype=GRB.BINARY, name=f"z_{j}_{i}_{g}_{gg}")

    for i in processors:
        y[i] = m.addVar(vtype=GRB.BINARY, name='y_'+i)
        L[i] = m.addVar(vtype=GRB.INTEGER, name='L_'+i)
        phi[i] = m.addVar(vtype=GRB.INTEGER, name='Phi_'+i)

    for g in orders:
        C[g] = m.addVar(vtype=GRB.INTEGER, name=g) 
        for gg in orders:
            xo[g,gg] = m.addVar(vtype=GRB.BINARY, name=f"xo_{g}_{gg}")

    m.update()

    # obj 
    alpha = 0.5
    m.setObjective(alpha * gp.quicksum(w[g] * C[g] for g in orders) + (1-alpha) * gp.quicksum(phi[i] for i in processors), sense=GRB.MINIMIZE)

    # constrains
    for j in jobs:
        m.addConstr(gp.quicksum(x[j,i] for i in processors) == 1)

    for j in jobs:
        for i in processors:
            m.addConstr(y[i] >= x[j,i])

    # for j in jobs:
    #     for i in processors:
    #         for l in range(1,len(jobs)):
    #             m.addConstr(S[i,l+1] - S[i,l] >= x[j,i,l] * t[j,i])
    

    for i in processors:
        m.addConstr(L[i] == gp.quicksum(x[j,i] * t[j,i] for j in jobs))
        m.addGenConstrPWL(L[i], phi[i], [0, 0, b[i], b[i]+1], [0, f[i], f[i], f[i]+u[i]], name='Phi_'+i)

    for g in orders:
        for gg in orders:
            if g != gg:
                m.addConstr(xo[g,gg] + xo[gg,g] == 1)
    for g in orders:
        for i in processors:
                m.addConstr(C[g] >= gp.quicksum(gp.quicksum(z[j,i,k,g] * t[j,i] for j in job_in_order[k]) for k in orders) + gp.quicksum(x[jj,i] * t[jj,i] for jj in job_in_order[g]))
    
    for i in processors:    
        for k in orders:
            for g in orders:
                if k != g:
                    for j in job_in_order[k]:
                        m.addConstr(z[j,i,k,g] >= 1 - (2-xo[k,g]-x[j,i]) * M)

    m.Params.TimeLimit = 1200
    m.optimize()
    m.write("work_schedule.lp")

    if m.status == GRB.Status.OPTIMAL:
        print('Got the optimal result')
    else:
        print('Not the optimal result')
    print ("current objective value is %g"%m.objVal)

    ## extract the result
    result = {}
    solution = m.getAttr('x', x)
    order_perm = m.getAttr('x', xo)
    order_time = m.getAttr('x', C)
    makespan = m.getAttr('x', L)
    aquisition_time = m.getAttr('x', phi)

    print('order permutation:')
    for g in orders:
        for gg in orders:
            print(order_perm[g,gg], end=' ')
        print()

    print('order completion time :', order_time)
    print('maskspan of each processors :', makespan)
    print('aquisition_time :', aquisition_time)

    res = defaultdict(list)
    for j in jobs:
        for i in processors:
            if solution[j,i] == 1:
                
                res['job'].append(j)
                res['processor'].append(i)
                res['process_time'].append(t[j,i])
                for k, v in job_in_order.items():
                    if j in v:
                        res['order'].append(k)
    # for i in processors:
    #     for l in range(1, len(jobs)+1):
    #         print(f'start time of pressoor {i} at posiotn {l} :', start_time[i,l])
    #     print('-------')

    print(pd.DataFrame(res))

    # draw gantt chart
    draw_gannt_chart(result, orders, makespan, image_name)

def draw_gannt_chart(result : dict[str, dict[str, Union[str, float]]], orders:list[str], makespan:dict[str, int], image_name:str) -> None:
    df = pd.DataFrame({ 'job': result.keys(),
                        'processor': [result[j]['processor'] for j in result],
                        'position': [result[j]['position'] for j in result],
                        'process_time': [result[j]['process_time'] for j in result],
                        'start_time': [ result[j]['start_time'] for j in result],
                        'order' : [ result[j]['order'] for j in result],
                        })
    
    fig, ax = plt.subplots()  
    barh_width = 2
    yticks = np.linspace(0, 10,num=len(processors)+2)[1:-1]
    xticks = np.arange(max(makespan.values()) + 1)
    colors = random.sample(cm.Accent.colors, len(orders))
    colors = {orders[i]:colors[i] for i in range(len(orders))}
    ax.set_ylim(0, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels(processors)
    ax.set_xticks(xticks)

    for i, p in enumerate(processors):
        barh = []
        facecolors = []
        job_h = []
        for index, row in df[df['processor'] == p].sort_values('start_time').iterrows():
            barh.append((row['start_time'], row['process_time']))
            job_h.append(row['job'])
            facecolors.append(colors[row['order']])
        # barh.sort()
        ax.broken_barh(xranges=barh, yrange=(yticks[i] - barh_width/2, barh_width), edgecolor='black', facecolor=facecolors)
        for (x1, x2), j in zip(barh, job_h):
                ax.text(x=x1 + x2/2, 
                        y=yticks[i],
                        s=j , 
                        ha='center', 
                        va='center',
                        color='black',
                    )


    # # ax.barh(y=df['processor'], width=df['process_time'], left=df['start_time'])
    # # for bar, disease in zip(ax.patches, jobs):
    #     # ax.text(0.1, bar.get_y()+bar.get_height()/2, 'test', color = 'white', ha = 'left', va = 'center') 
    fig.savefig(image_name)


jobs = [8,5,7,8,4,6,6,5]

orders = {
    'o1' : [{'j2','j4','j5'}, 10],
    'o2' : [{'j3','j6'}, 8], 
    'o3' : [{'j1','j7','j8'}, 7]
}

# unit cost, purchased cost, base time, speed
processors = {
    "p1" : [25,15,10,2.1],
    "p2" : [15,10,15,1.5],
}

#order_delivery_ip(jobs, orders, processors, '8jobs_3orders_2processors.png')


jobs = [4,5,9,10,4,10,6,5,2,3,12,6]

orders = {
    'o1' : [{'j4','j5','j11'}, 3],
    'o2' : [{'j3','j6'}, 4], 
    'o3' : [{'j1','j7','j8', 'j10'}, 5],
    'o4' : [{'j2','j9', 'j12'}, 6]
}

processors = {
    "p1" : [25,15,10,1.6],
    "p2" : [15,10,15,1.3],
    "p3" : [20,12,14,1.7],
}

order_delivery_ip(jobs, orders, processors, '12jobs_4orders_3processors.png')


jobs = [4,5,9,10,4,10,6,5,2,3,12,6,5,15,10]

orders = {
    'o1' : [{'j4','j5','j11'}, 3],
    'o2' : [{'j3','j6'}, 4], 
    'o3' : [{'j1','j7','j8', 'j10'}, 5],
    'o4' : [{'j2','j9', 'j12'}, 6],
    'o5' : [{'j13','j14', 'j15'}, 4]
}

processors = {
    "p1" : [25,15,10,1.6],
    "p2" : [15,10,15,1.3],
    "p3" : [20,12,14,1.7],
    "p4" : [18,11,12,1.9]
}

# order_delivery_ip(jobs, orders, processors, '15jobs_5orders_4processors.png')


# # job computing load
# jobs = [8,5,7,8,4,6,6,5]
# jobs = {'j'+str(j+1) : jobs[j] for j in range(len(jobs))}

# # Order 
# orders, job_in_order, w = gp.multidict({
#     'o1' : [{'j2','j4','j5'}, 10],
#     'o2' : [{'j3','j6'}, 8],
#     'o3' : [{'j1','j7','j8'}, 7]
# })

# # processor
# processors, u, f, b, s = gp.multidict({
#     "p1" : [25,15,10,1.6],
#     "p2" : [15,10,15,1.3],
# })

# # job process time in each processor
# t = {(j,p) : math.ceil(jl/ss) for j, jl in jobs.items() for p, ss in s.items()}

# m = gp.Model("parallel_order")

# M = 1000000
# x = {}
# y = {}
# # C = {}
# S = {}
# L = {}
# O = {}
# phi = {}
# # decision variables
# for j in jobs:
#     for i in processors:
#         for l in range(1,len(jobs)+1):
#             x[j,i,l] = m.addVar(vtype=GRB.BINARY, name=f"x_{j}_{i}_{l}")

# for i in processors:
#     y[i] = m.addVar(vtype=GRB.BINARY, name='y_'+i)
#     L[i] = m.addVar(vtype=GRB.INTEGER, name='L_'+i)
#     phi[i] = m.addVar(vtype=GRB.INTEGER, name='Phi_'+i)

# # for j in jobs:
#     # C[j] = m.addVar(vtype=GRB.INTEGER, name=f"c_{j}")

# for i in processors:
#     for l in range(1,len(jobs)+1):
#         S[i,l] = m.addVar(vtype=GRB.INTEGER, name=f"s_{i}_{l}")

# for o in orders:
#     O[o] = m.addVar(vtype=GRB.INTEGER, name=o) 

# m.update()

# # obj 
# alpha = 0.5
# m.setObjective(alpha * gp.quicksum(w[o] * O[o] for o in orders) + (1-alpha) * gp.quicksum(phi[i] for i in processors), sense=GRB.MINIMIZE)

# # constrains
# for j in jobs:
#     m.addConstr(gp.quicksum(gp.quicksum(x[j,i,l] for l in range(1,len(jobs)+1)) for i in processors) == 1)
#     # m.addConstr(C[j] >= 0)

# for j in jobs:
#     for i in processors:
#         for l in range(1,len(jobs)+1):
#             # m.addConstr(C[j] >=  x[j,i,l] * (S[i,l] + t[j,i]))
#             # m.addConstr(C[j] >= S[i,l] + x[j,i,l] * t[j,i] - (1-x[j,i,l]) * M)
#             m.addConstr(y[i] >= x[j,i,l])

# for j in jobs:
#     for i in processors:
#         for l in range(1,len(jobs)):
#             m.addConstr(S[i,l+1] - S[i,l] >= x[j,i,l] * t[j,i])

# for i in processors:
#     for l in range(1, len(jobs)):
#         m.addConstr(gp.quicksum(x[j,i,l] - x[j,i,l+1] for j in jobs) >= 0)

# for i in processors:
#     for l in range(1,len(jobs)+1):
#         m.addConstr(gp.quicksum(x[j,i,l] for j in jobs) <= 1)

# for i in processors:
#     m.addConstr(L[i] == gp.quicksum(gp.quicksum(x[j,i,l] * t[j,i] for l in range(1, len(jobs)+1)) for j in jobs))
#     m.addGenConstrPWL(L[i], phi[i], [0, 0, b[i], b[i]+1], [0, f[i], f[i], f[i]+u[i]], name='Phi_'+i)

# for g in orders:
#     for j in job_in_order[g]:
#         m.addConstr(O[g] >= gp.quicksum(gp.quicksum(x[j,i,l] * (S[i,l] + t[j,i]) for l in range(1,len(jobs)+1)) for i in processors), name=f'job:{j} in order:{g}')

# #m.Params.TimeLimit = 5
# m.optimize()
# m.write("work_schedule.lp")

# print('jons preocess time: ', t)

# # if m.status == GRB.Status.OPTIMAL:
# print ("current objective value is %g"%m.objVal)

# result = defaultdict(list)
# solution = m.getAttr('x', x)
# # ctime = m.getAttr('x', C)
# start_time = m.getAttr('x', S)
# order_time = m.getAttr('x', O)
# makespan = m.getAttr('x', L)
# aquisition_time = m.getAttr('x', phi)
# # print('ctime :', ctime)
# print('order time :', order_time)
# print('maskspan :', makespan)
# print('aquisition_time :', aquisition_time)



# for j in jobs:
#     for i in processors:
#         for l in range(1, len(jobs)+1):
#             if solution[j,i,l] == 1:
#                 result[j].append(i) 
#                 result[j].append(l)
#                 result[j].append(t[j,i])
#                 result[j].append(start_time[i,l])
#                 for k, v in job_in_order.items():
#                     if j in v:
#                         result[j].append(k)
#                 print(f'job{j} is schedule at processor{i} in position{l}, start time of {start_time[i,l]}')

# for i in processors:
#     for l in range(1, len(jobs)+1):
#         print(f'start time of pressoor {i} at posiotn {l} :', start_time[i,l])
#     print('-------')

# print(result)
# # start_time = m.getAttr('x', s)
# # for i in processors:
# #     for l in range(1, len(jobs)+1):
# #         print(f'start time of processor {i} at posistion {l} :', s[i,l])

# # draw gantt chart
# df = pd.DataFrame({'job': jobs.keys(),
#                   'processor': [result[j][0] for j in jobs],
#                   'position': [result[j][1] for j in jobs],
#                   'process_time': [result[j][2] for j in jobs],
#                   'start_time': [ result[j][3] for j in jobs],
#                   'order' : [ result[j][4] for j in jobs],
#                 })
# print(df)


# fig, ax = plt.subplots()  

# barh_width = 2
# yticks = np.linspace(0, 10,num=len(processors)+2)[1:-1]
# xticks = np.arange(20)
# cmap = matplotlib.colormaps['Accent']  # type: matplotlib.colors.ListedColormap
# colors = random.sample(cm.Accent.colors, len(orders))
# colors = {orders[i]:colors[i] for i in range(len(orders))}
# ax.set_ylim(0, 10)
# ax.set_yticks(yticks)
# ax.set_yticklabels(processors)
# ax.set_xticks(xticks)

# for i, p in enumerate(processors):
#     barh = []
#     facecolors = []
#     job_h = []
#     for index, row in df[df['processor'] == p].sort_values('start_time').iterrows():
#         barh.append((row['start_time'], row['process_time']))
#         job_h.append(row['job'])
#         facecolors.append(colors[row['order']])
#     # barh.sort()
#     ax.broken_barh(xranges=barh, yrange=(yticks[i] - barh_width/2, barh_width), edgecolor='black', facecolor=facecolors)
#     for (x1, x2), j in zip(barh, job_h):
#             ax.text(x=x1 + x2/2, 
#                     y=yticks[i],
#                     s=j , 
#                     ha='center', 
#                     va='center',
#                     color='white',
#                    )


# # # ax.barh(y=df['processor'], width=df['process_time'], left=df['start_time'])
# # # for bar, disease in zip(ax.patches, jobs):
# #     # ax.text(0.1, bar.get_y()+bar.get_height()/2, 'test', color = 'white', ha = 'left', va = 'center') 
# fig.savefig('test2.png')
