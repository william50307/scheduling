from collections import defaultdict
from functools import reduce
import math
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from tqdm import tqdm
import pathlib
from utils.data_generation import DataGenerator
from utils.util import args_parser, get_data
from ip_w_th2 import order_delivery_ip
from utils.components import Order, OrderPerm, Processor, CostCalculator, Solution


class Heuristic():
    def __init__(self, 
                #  orders_info:dict[str:dict[str:set, str:int]], 
                #  processors_info:dict[str:list[float]],
                #  jobs_load:dict[str:str],
                #  jobs_order:dict[str:str],
                 helper_data:dict[str:dict],
                 alpha:float=0.5):
        
        self.alpha = alpha
        self.helper_data = helper_data
        self.orders = {}
        for k, v in helper_data['orders_info'].items():
            self.orders[k] = Order(v['jobs'], v['weight'], k, helper_data['jobs_load'])

        self.processors = {}
        for k, v in helper_data['processors_info'].items():
            self.processors[k] = Processor(k, *v, helper_data['jobs_load'], helper_data['jobs_order'])
        
        self.cost_calculator = CostCalculator(helper_data['jobs_load'], helper_data['orders_info'])


    '''
    list scheduling:
    select the processor which has earliest completion time
    '''
    def ls(self, order_perm_type:str, job_perm_type:str) -> tuple[set[Processor], list[str], int]:
        order_perm = OrderPerm.get_order_perm(self.orders, order_perm_type)
        jobs = []
        for o in order_perm:
            self.orders[o].set_job_mode(job_perm_type)
            jobs += self.orders[o].jobs

        # choose the processor that can finished the order at the earliest time s
        for j in jobs:
            f_time = float('inf')
            for p in self.processors:
                processors = copy.deepcopy(self.processors)
                processors[p].add(j)
                t = self.cost_calculator.order_finish_time(processors, self.helper_data['jobs_order'][j], order_perm)
                if t < f_time:
                    f_time = t
                    temp_p = p
            self.processors[temp_p].add(j)
        

        return Solution(self.processors, order_perm, self.cost_calculator, self.alpha)
        

    def fbs(self, order_perm_type:str, job_perm_type:str) -> tuple[set[Processor], list[str], int]:

        order_perm = OrderPerm.get_order_perm(self.orders, order_perm_type)
        fbs_list = sorted(self.processors.values(), key=lambda p:p.f/p.b/p.s)
        p = fbs_list.pop(0)
        purchased_p = {p}

        for o in order_perm:
            # set job permutation
            self.orders[o].set_job_mode(job_perm_type)
            p.add_order(self.orders[o])

        while fbs_list:
            can_p = fbs_list.pop(0)
            cur_cost = self.cost_calculator.cost(purchased_p, order_perm, self.alpha)
            min_cost = float('inf')
            flag = False
            while True:
                # find the minimun cost after switch one job to another
                for p in purchased_p:
                    for j in p.get_all_jobs():
                        simu_p = copy.deepcopy(p)
                        # print(simu_p.jobs)
                        simu_p.remove(j)
                        can_p.add(j)
                        temp_p = purchased_p.copy()
                        temp_p.remove(p)
                        temp_p.update((simu_p, can_p))
                        #print('current job:', j)
                        #can_p.print_jobs()
                        cost = self.cost_calculator.cost(temp_p, order_perm, self.alpha)
                        if cost < min_cost:
                            job = j
                            old_p = p
                            new_p = simu_p
                            min_cost = cost
                        can_p.remove(j)
                        
                if min_cost < cur_cost:
                    flag = True
                    # add job to can_p 
                    can_p.add(job)
                    purchased_p.add(new_p)
                    purchased_p.remove(old_p)
                    cur_cost = min_cost
                else:
                    if flag:
                        purchased_p.add(can_p)
                        break
                    else:
                        return Solution({p.id : p for p in purchased_p}, order_perm, self.cost_calculator, self.alpha)
        
        # return purchased_p, order_perm, cur_cost
        return Solution({p.id : p for p in purchased_p}, order_perm, self.cost_calculator, self.alpha)
            
    def bf(self, order_perm_type:list[str], job_perm_type:list[str]) -> Solution:
        order_perm = OrderPerm.get_order_perm(self.orders, order_perm_type)
        jobs = defaultdict(list)
        jobs_seq = []
        for o in order_perm:
            self.orders[o].set_job_mode(job_perm_type)
            jobs[o] = self.orders[o].jobs.copy()
        
        while reduce(lambda x,y:x+y , jobs.values()) != []:
            for o in order_perm:
                if jobs[o] != []:
                    jobs_seq.append(jobs[o].pop(0))
    
        assert set(jobs_seq) == set(self.helper_data['jobs_load'].keys()), 'missing some jobs'
        
        purchased_p = set()
        fbs_list = sorted(self.processors.values(), key=lambda p:p.f/p.b/p.s)
        while jobs_seq and fbs_list:
            p = fbs_list.pop(0)
            while p.get_makespan() < p.b and jobs_seq:
                p.add(jobs_seq.pop(0))
            purchased_p.add(p)

        if not jobs_seq:
            cur_sol = Solution({p.id : p for p in purchased_p}, order_perm, self.cost_calculator, self.alpha)
            while True:
                min_cost = cur_sol.getCost()
                for pair in cur_sol.get_all_swap_pairs():
                    new_sol = Tabusearch.swap_job(cur_sol, pair)
                    if new_sol.getCost() < min_cost:
                        min_cost = new_sol.getCost() 
                        can_sol = new_sol
                
                for pair in cur_sol.get_all_insert_pairs():
                    new_sol = Tabusearch.insert_job(cur_sol, pair)
                    if new_sol.getCost() < min_cost:
                        min_cost = new_sol.getCost()
                        can_sol = new_sol
                
                if min_cost < cur_sol.getCost():
                    cur_sol = can_sol
                else:
                    return cur_sol
        else:
            # find a best processors to place
            while jobs_seq:
                j = jobs_seq.pop()
                min_cost = float('inf')
                for p in purchased_p:
                    simu_p = copy.deepcopy(p)
                    simu_p.add(j)
                    temp_p = purchased_p.copy()
                    temp_p.remove(p)
                    temp_p.add(simu_p)
                    cost = self.cost_caculator.cost(temp_p, order_perm, self.alpha)
                    if cost < min_cost:
                        min_cost = cost
                        can_p = p 
                can_p.add(j)
            return Solution({p.id : p for p in purchased_p}, order_perm, self.cost_calculator, self.alpha)

class TabuList():
    def __init__(self, tabu_len=7):
        self.tabu_list = []
        self.tabu_len = tabu_len

    def add(self, new_sol):
        if len(self.tabu_list) == self.tabu_len:
            self.tabu_list.pop(0)
            self.tabu_list.append(new_sol)
    
    def __contains__(self, item):
        return item in self.tabu_list

class Tabusearch():
    def __init__(self, current_solution:Solution, tabu_list:TabuList, alpha:int=None, iters:int=100):
        self.sol = current_solution
        self.alpha = alpha
        self.iters = iters
        self.tabu_list = tabu_list

    '''swap two jobs'''
    @staticmethod
    def swap_job(sol:Solution, pair:tuple[tuple[str,str]]) -> Solution:
        (p1,j1), (p2,j2) = pair
        new_sol = copy.deepcopy(sol)
        new_sol.processors[p1].remove(j1)
        new_sol.processors[p1].add(j2)
        new_sol.processors[p2].remove(j2)
        new_sol.processors[p2].add(j1)
        return new_sol
    
    '''remove a job from processor and add to another'''
    @staticmethod
    def insert_job(sol, pair:tuple[str,str,str]) -> Solution:
        j, p1, p2 = pair
        new_sol = copy.deepcopy(sol)
        new_sol.processors[p1].remove(j)
        new_sol.processors[p2].add(j)
        return new_sol     

    def get_neighbors(self, sol:Solution, k:float) -> list[Solution]:
        res = []
        thr = random.random() * 100
        if thr <= k:
            # switch order permutation
            random.shuffle(self.sol.orders_perm)
        '''job swap'''
        for pair in sol.get_all_swap_pairs():
            new_sol = self.swap_job(sol, pair)
            res.append(new_sol)
        '''Insert'''
        for pair in sol.get_all_insert_pairs():
            new_sol = self.insert_job(sol, pair)
            res.append(new_sol)

        return res

    def local_search(self, current_sol:Solution, k:float, tabu_check:bool=True) -> Solution:
        neighbors = self.get_neighbors(current_sol, k)  

        min_cost = float('inf')
        best_neighbor = None
        for n in neighbors:
            if n.getCost() < min_cost and (not tabu_check or n not in self.tabu_list):
                min_cost =  n.getCost()
                best_neighbor = n
        
        return best_neighbor
    
    def vnts(self, tabu_check=True) -> Solution:
        order_perm_num = 100
        best_sol = self.sol
        current_sol = self.sol
        k_max = order_perm_num
        k = 0
        while k < k_max:
            best_neighbor = self.local_search(copy.deepcopy(current_sol), k, tabu_check)
            if best_neighbor.getCost() < best_sol.getCost():
                best_sol = best_neighbor
                current_sol = best_neighbor
                self.tabu_list.add(best_neighbor)
                k = 0
            else:
                k += 1        
        return best_sol
        
    
    def tabu_search(self) -> Solution:
        best_sol = self.sol
        current_sol = self.sol
 

        for _ in range(self.iters):
            neighbors = self.get_neighbors(current_sol)
            min_cost = float('inf')
            best_neighbor = None
            for sol in neighbors:
                c = sol.getCost()
                if sol not in self.tabu_list and c < min_cost:
                    best_neighbor = sol
                    min_cost = c

            if best_neighbor is None:
                print('No non-tabu neighbors found')
                return best_sol
            
            current_sol = best_neighbor
            self.tabu_list.add(best_sol)
    
            if best_neighbor.getCost() < best_sol.getCost():
                best_sol = best_neighbor
        print(best_sol.getCost())
        return best_sol
    
def draw_gannt_chart(sol:Solution, image_name:str, title:str) -> None:
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
    colors = random.sample(cm.Accent.colors, len(sol.orders_perm))
    colors = {sol.orders_perm[i]:colors[i] for i in range(len(sol.orders_perm))}
    ax.set_title(title + f'cost : {sol.getCost()}')
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


if __name__ == '__main__':
 
    args = args_parser()
    random.seed(args.seed)
    np.random.seed(args.seed)


    jobs_load, orders_info, processors_info, jobs_order = get_data(args, DataGenerator)
    min_calculate = defaultdict(lambda:0)

    for i in tqdm(range(args.instance_num)):
        orders_perm = ['ml', 'wml', 'wp']
        jobs_perm = ['slf', 'llf']
        draw = True if i == 0 else False
        saving_prefix = args.saving_folder + '/' + f'{args.job_num}_{args.order_num}_{args.processor_num}'
        pathlib.Path(saving_prefix).mkdir(parents=True, exist_ok=True) 
        res = defaultdict(list) 
        for op in orders_perm:
            for jp in jobs_perm:
                orders, processors = op_init(orders_info, processors_info)
                heuristic = Heuristic(orders, processors, args.alpha)
                sol = heuristic.fbs(op, jp)
                if draw:
                    draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_fbs.png', '')
                res[(op, jp, 'fbs')].append(sol.getCost())

                # tabulist = TabuList()
                # tabusearch = Tabusearch(sol, tabulist, 0.3)
                # sol = tabusearch.vnts()
                # if draw:
                #     draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_fbs_tabuSearch.png', '')

                orders, processors = op_init(orders_info, processors_info)
                heuristic = Heuristic(orders, processors, args.alpha)
                sol = heuristic.ls(op, jp)
                if draw:
                    draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_ls.png', '')
                res[(op, jp, 'ls')].append(sol.getCost())

                # tabulist = TabuList()
                # tabusearch = Tabusearch(sol, tabulist, 0.3)
                # sol = tabusearch.vnts()
                # if draw:
                #     draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_ls_tabuSearch.png', '')

                orders, processors = op_init(orders_info, processors_info)
                heuristic = Heuristic(orders, processors, args.alpha)
                sol = heuristic.bf(op, jp)
                if draw:
                    draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_bf.png', '')
                res[(op, jp, 'bf')].append(sol.getCost())

                # tabulist = TabuList()
                # tabusearch = Tabusearch(sol, tabulist, 0.3)
                # sol = tabusearch.vnts()
                # if draw:
                #     draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_bf_tabuSearch.png', '')

        # min_calculate[min(res, key=res.get)] += 100 / args.instance_num
                    
    # print(min_calculate)
    # for k in res:
    #     print(f'{k} mean csot :', sum(res[k]) / len(res[k]))

    # # temp code
    # # print('start ip modeling')
    # orders, processors = op_init(orders_info, processors_info)
    # jobs = list(jobs_load.values())
    # orders = {k:[v['jobs'], v['weight']] for k,v in orders_info.items()}
    # processors = processors_info

    # order_delivery_ip(jobs, orders, processors, args.alpha, '20jobs_5orders_3processors.png')