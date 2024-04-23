
from collections import defaultdict
import math
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from utils.components import Solution
import matplotlib.colors as mcolors

def draw_gannt_chart(jobs_load:dict[str:int], sol:Solution, image_name:str, title:str) -> None:
    infos = defaultdict(list)
    for p in sol.processors.values():
        start_time = 0
        for o in sol.orders_perm:
            for j in sorted(p.jobs[o]):    
                infos['job'].append(j)
                infos['processor'].append(p.id)
                infos['process_time'].append(math.ceil(jobs_load[j] / p.s))
                infos['start_time'].append(start_time)
                infos['order'].append(o)
                start_time +=  math.ceil(jobs_load[j] / p.s)

    df = pd.DataFrame(infos)
    
    fig, ax = plt.subplots()  
    barh_width = 2
    yticks = np.linspace(0, 10,num=len(sol.processors)+2)[1:-1]
    xticks = np.arange(max([p.get_makespan() for p in sol.processors.values()]) + 1)
    colors = list(mcolors.TABLEAU_COLORS.values())
    colors = {sol.orders_perm[i]:colors[i] for i in range(len(sol.orders_perm))}
    ax.set_title(title + '\n' + f'cost : {sol.getCost()}')
    ax.set_ylim(0, 10)
    ax.set_yticks(yticks)
    ax.set_yticklabels([p.id for p in sol.processors.values()])
    ax.set_xticks(xticks)

    for i, p in enumerate(sol.processors.values()):
        barh = []
        facecolors = []
        job_h = []
        for _, row in df[df['processor'] == p.id].sort_values('start_time').iterrows():
            barh.append((row['start_time'], row['process_time']))
            job_h.append(row['job'])
            facecolors.append(colors[row['order']])
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