import pandas as pd 
import numpy as np 
import random

import matplotlib.pyplot as plt
import matplotlib.cm as cm

def draw_gannt_chart_ip(result : dict[str, dict[str, str|float]], orders:list[str], makespan:dict[str, int], image_name:str, processors) -> None:
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
        for _, row in df[df['processor'] == p].sort_values('start_time').iterrows():
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