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
from loguru import logger
import matplotlib.colors as mcolors

def order_delivery_ip(jobs : list[int], orders:dict[str,list[set , int]], processor_info:dict[str, list[float]], alpha:float, image_name:str) -> tuple[bool, float]:

    orders = {k:[v['jobs'], v['weight']] for k,v in orders.items()}
    orders, job_in_order, w = gp.multidict(orders)
    processors, b, u, f, s = gp.multidict(processor_info)

    # job process time in each processor
    t = {(j,p) : math.ceil(jl/ss) for j, jl in jobs.items() for p, ss in s.items()}

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
                # z[j,i,g] = m.addVar(vtype=GRB.BINARY, name=f"z_{j}_{i}_{g}")
                    z[j,i,g,gg] = m.addVar(vtype=GRB.BINARY, name=f"z_{j}_{i}_{g}_{gg}")

    for i in processors:
        y[i] = m.addVar(vtype=GRB.BINARY, name='y_'+i)
        L[i] = m.addVar(vtype=GRB.INTEGER, name='L_'+i)
        phi[i] = m.addVar(name='Phi_'+i)

    for g in orders:
        C[g] = m.addVar(vtype=GRB.INTEGER, name=g) 
        for gg in orders:
            xo[g,gg] = m.addVar(vtype=GRB.BINARY, name=f"xo_{g}_{gg}")

    m.update()

    # obj 
    m.setObjective(alpha * gp.quicksum(w[g] * C[g] for g in orders) + (1-alpha) * gp.quicksum(phi[i] for i in processors), sense=GRB.MINIMIZE)

    # constrains
    for j in jobs:
        m.addConstr(gp.quicksum(x[j,i] for i in processors) == 1)

    for j in jobs:
        for i in processors:
            m.addConstr(y[i] >= x[j,i])


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
                #m.addConstr(C[g] >= gp.quicksum(gp.quicksum(z[j,i,g] * t[j,i] for j in job_in_order[k]) for k in orders if k!=g) + gp.quicksum(x[jj,i] * t[jj,i] for jj in job_in_order[g]))
    
    for i in processors:    
        for k in orders:
            for g in orders:
                if k != g:
                    for j in job_in_order[k]:
                        m.addConstr(z[j,i,k,g] >= 1 - (2-xo[k,g]-x[j,i]) * M)
                        #m.addConstr(z[j,i,g] >= 1 - (2-xo[k,g]-x[j,i]) * M)

    m.Params.TimeLimit = 6000
    m.optimize()
    m.write("work_schedule.lp")

    if m.status == GRB.Status.OPTIMAL:
        logger.info('Got the optimal result')
    else:
        logger.info('Not the optimal result')
    logger.info("current objective value is %g" %m.objVal)

    # extract the result
    solution = m.getAttr('x', x)
    order_perm = m.getAttr('x', xo)
    order_time = m.getAttr('x', C)
    makespan = m.getAttr('x', L)
    aquisition_cost = m.getAttr('x', phi)

    order_seq = [0] * len(orders)

    # get order permutation
    for g in orders:
        s = 0
        for gg in orders:
            s += order_perm[g,gg]
        order_seq[int(len(orders)-1-s)] = g
    
    logger.debug(f'order completion time : {order_time}')
    logger.debug(f'maskspan of each processors : {makespan}')
    logger.debug(f'aquisition_cost : {aquisition_cost}')
    logger.debug(f'order permutation : {order_seq}')

    res = defaultdict(list)
    processor_start_time = {p:0 for p in processors}
    for j in jobs:
        for i in processors:
            if solution[j,i] == 1:
                res['job'].append(j)
                res['process_time'].append(t[j,i])
                res['processor'].append(i)
                processor_start_time[i] += res['process_time'][-1]
                for k, v in job_in_order.items():
                    if j in v:
                        res['order'].append(k)
    
    start_time = [0] * len(res['job'])
    for i in processors:
        p_start = 0
        for o in order_seq:
            for idx,(p,oo) in enumerate(zip(res['processor'], res['order'])):
                if i == p and o == oo:
                    start_time[idx] = p_start
                    p_start += res['process_time'][idx]

    res['start_time'] = start_time

    logger.debug(pd.DataFrame(res))

    # draw gantt chart
    draw_gannt_chart_ip(pd.DataFrame(res), orders, makespan, image_name, m.objVal, processors)

    return m.status == GRB.Status.OPTIMAL, m.objVal

def draw_gannt_chart_ip(df : dict[str, dict[str, Union[str, float]]], orders:list[str], makespan:dict[str, int], image_name:str, cost, processors) -> None:
    # df = pd.DataFrame({ 'job': result.keys(),
    #                     'processor': [result[j]['processor'] for j in result],
    #                     'position': [result[j]['position'] for j in result],
    #                     'process_time': [result[j]['process_time'] for j in result],
    #                     'start_time': [ result[j]['start_time'] for j in result],
    #                     'order' : [ result[j]['order'] for j in result],
    #                     })
    
    fig_size = 10
    fig, ax = plt.subplots()  
    barh_width = fig_size // (len(processors) + 1)
    yticks = np.linspace(0, fig_size,num=len(processors)+2)[1:-1]
    xticks = np.arange(max(makespan.values()) + 1)
    # colors = random.sample(cm.Accent.colors, len(orders))
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors = {orders[i]:colors[i] for i in range(len(orders))}
    ax.set_title(f'cost : {cost}')
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

    fig.savefig(image_name)

if __name__ == "__main__":
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
