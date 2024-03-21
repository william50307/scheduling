from collections import defaultdict
from functools import reduce
import math
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from itertools import combinations
from tqdm import tqdm
import pathlib
from utils.data_generation import DataGenerator
from utils.util import args_parser, get_data
from ip_w_th2 import order_delivery_ip


class Order():
    def __init__(self, jobs:list[str], weight:int, order_id:str):
        self.id = order_id
        self.weight = weight
        self.jobs = jobs
        self.jobs_slf = sorted(jobs, key=lambda x:jobs_load[x])
        self.jobs_llf = sorted(jobs, key=lambda x:jobs_load[x], reverse=True)
        self.job_mode = None 

    def set_job_mode(self, job_mode:str) -> None:
        if job_mode not in ['slf', 'llf']:
            print('not a valid job mode for order')
            exit()
        self.job_mode = job_mode
        if self.job_mode == 'slf':
            self.jobs = self.jobs_slf
        elif self.job_mode == 'llf':
            self.jobs = self.jobs_llf

    def sum_load(self) -> int:
        return sum(jobs_load[j] for j in self.jobs)

    def __str__(self):
        return self.id
    
class Processor():
    def __init__(self, id, base_time, unit_cost, fixed_charge, speed):
        self.id = id
        self.b = base_time
        self.c = unit_cost
        self.f = fixed_charge
        self.s = speed 
        self.jobs = defaultdict(set)

    '''add single job into processor'''
    def add(self, job:str) -> None:
        self.jobs[jobs_order[job]].add(job)

    '''remove single job from processor'''
    def remove(self, job:str) -> None:
        self.jobs[jobs_order[job]].remove(job)

    def get_all_jobs(self) -> set:
        return set(reduce(lambda x,y:x.union(y), self.jobs.values()))

    '''add all jobs in that order into processor'''
    def add_order(self, order:Order) -> None:
        self.jobs[order.id] = self.jobs[order.id].union(order.jobs)

    '''get cuurent maskspan'''
    def get_makesapn(self) -> int:
        global jobs_load
        return sum(math.ceil(jobs_load[j] / self.s) for s in self.jobs.values() for j in s)

    '''the processors total cost'''
    def cost(self) -> int:
        return self.f + self.c * max(0, self.get_makesapn() - self.b)
    
    '''get order finished time in this processor'''
    def getOrderFinishedTimeinP(self, order_id:str, order_perm:list[str]) -> int:
        global jobs_load
        process_time = 0
        for o in order_perm:
            for j in self.jobs[o]:
                process_time += math.ceil(jobs_load[j] / self.s)
            if o == order_id:
                return process_time
        
    '''print the job orders'''
    def print_jobs(self) -> None:
        for o in self.jobs:
            print(o, ":")
            jobs = list(self.jobs[o])
            print(' -> '.join(jobs))
        return

class OrderPerm():
    '''
    the orders are in the decreasing order of weight (w_g)
    '''
    @staticmethod
    def weight_priority(orders) -> list[str]:
        return [o.id for o in sorted(orders.values(), key=lambda x:x.weight, reverse=True)]

    '''
    the orders are in ascending order of summation of all jobs load in each order
    '''
    @staticmethod
    def minimum_load(orders) -> list[str]:
        return [o.id for o in sorted(orders.values(), key=lambda x:x.sum_load())]

    '''
    the orders are ordered in the non-decreasing order of wg/sum(load )
    '''
    @staticmethod
    def weight_minimum_load(orders) -> list[str]:
        return [o.id for o in sorted(orders.values(), key=lambda x:x.weight / x.sum_load())]

    @classmethod
    def get_order_perm(cls, orders, mode) -> list[str]:
        if mode == 'wp':
            return cls.weight_priority(orders)
        elif mode == 'ml':
            return cls.minimum_load(orders)
        elif mode == 'wml':
            return cls.weight_minimum_load(orders)
        else:
            print(f'ERROR : there is no mode named {mode} in order permutation')
            exit()

class Solution():
    def __init__(self, processors:dict[str:Processor], orders_perm:list[str]):
        self.processors = processors
        self.orders_perm = orders_perm

    def getCost(self):
        return CostCalculator.cost(set(self.processors.values()), self.orders_perm)

    ''' used on tabu sarch, find all the job pairs which can be swaped'''
    def get_all_swap_pairs(self) -> list[tuple[tuple[str,str]]]:
        res = []
        orders = orders_info.keys()
        for p1, p2 in combinations(self.processors.keys(), 2):
            for o in orders:
                for j in self.processors[p1].jobs[o]:
                    for jj in self.processors[p2].jobs[o]:
                        res.append(((p1,j),(p2,jj))) 
        return res

    '''
    used on tabu search, find all the job pairs which can be inserted
    return value is a tuple contains three elements : [job name, processor remove from, processor insert into]
    '''
    def get_all_insert_pairs(self) -> list[tuple[str, str, str]]:
        res = []
        for p1, p2 in combinations(self.processors.keys(), 2):
            for o in orders:
                for j in self.processors[p1].jobs[o]:
                    res.append((j, p1, p2))
                for j in self.processors[p2].jobs[o]:
                    res.append((j, p2, p1))
        return res
                

class Heuristic():
    def __init__(self, orders:dict[str:Order], processors:dict[str:Processor]):
        self.orders = orders
        self.processors = processors
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
                t = CostCalculator.order_finish_time(processors, jobs_order[j], order_perm)
                if t < f_time:
                    f_time = t
                    temp_p = p
            self.processors[temp_p].add(j)
        
        cost = CostCalculator.cost(set(self.processors.values()), order_perm)
        # return set(self.processors.values()), order_perm, cost
        return Solution(self.processors, order_perm)
        

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
            cur_cost = CostCalculator.cost(purchased_p, order_perm)
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
                        cost = CostCalculator.cost(temp_p, order_perm)
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
                        return Solution({p.id : p for p in purchased_p}, order_perm)
        
        # return purchased_p, order_perm, cur_cost
        return Solution({p.id : p for p in purchased_p}, order_perm)
                
    def bf(self, order_perm_type, job_perm_type):
        order_perm = OrderPerm.get_order_perm(self.orders, order_perm_type)
        jobs = []
        for o in order_perm:
            self.orders[o].set_job_mode(job_perm_type)
            jobs += self.orders[o].jobs

        fbs_list = sorted(self.processors, key=lambda p:p.f/p.b/p.s)
        p = fbs_list.pop(0)
        purchased_p = {p}
        
        while True:
            for p in purchased_p:
                while p.get_makesapn() < p.base_time and jobs:
                    p.add(jobs.pop(0))
            if not jobs or not fbs_list:
                # return purchased_p, CostCalculator.cost(purchased_p, order_perm)
                return Solution({p.id : p for p in purchased_p}, order_perm)
            new_p = fbs_list.pop(0)
            while True:                
                cur_cost = CostCalculator.cost(purchased_p ,order_perm)
                # find if there is switch can reduce the total cost
                for p in purchased_p:   
                    for j in p.get_all_jobs():
                        new_p.add(j)
                        p.remove(j)
                        simu_set = copy.deepcopy(purchased_p)
                        simu_set.add(new_p)
                        cost = CostCalculator.cost(simu_set, order_perm)
                        p.add(j)
                        if cost < cur_cost:
                            cur_cost = cost
                            to_remove_p = p
                            can_j = j
                
                if can_j:
                    new_p.add(can_j)
                    to_remove_p.remove(can_j)
                    purchased_p.add(new_p)
                else:
                    if not new_p.get_all_jobs():
                        # return purchased_p, cost
                        return Solution({p.id : p for p in purchased_p}, order_perm)
                    else:
                        break

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
    def swap_job(self, sol:Solution, pair:tuple[tuple[str,str]]) -> Solution:
        (p1,j1), (p2,j2) = pair
        new_sol = copy.deepcopy(sol)
        new_sol.processors[p1].remove(j1)
        new_sol.processors[p1].add(j2)
        new_sol.processors[p2].remove(j2)
        new_sol.processors[p2].add(j1)
        return new_sol
    
    '''remove a job from processor and add to another'''
    def insert_job(self, sol, pair:tuple[str,str,str]) -> Solution:
        j, p1, p2 = pair
        new_sol = copy.deepcopy(sol)
        new_sol.processors[p1].remove(j)
        new_sol.processors[p2].add(j)
        return new_sol     

    def get_neighbors(self, sol:Solution, k:float) -> list[Solution]:
        res = []
        thr = random.random() * 100
        if thr <= k:
            random.shuffle(self.sol.orders_perm)
            # switch order permutation
        # job swap
        # if self.alpha is None or thr < self.alpha:
        for pair in sol.get_all_swap_pairs():
            new_sol = self.swap_job(sol, pair)
            res.append(new_sol)
        # Insert
        # if self.alpha is None or thr >= self.alpha:
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
 

        for _ in tqdm(range(self.iters)):
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
    
        
class CostCalculator():
    @staticmethod
    def cost(processors:set[Processor], order_perm:list[str]) -> int:
        finished_time = {}
        total_cost = 0
        for p in processors:
            fin = {}
            prev = 0
            for o in order_perm:
                fin[o] = prev + sum([math.ceil(jobs_load[j] / p.s) for j in p.jobs[o]])
                prev = fin[o]
            finished_time[p.id] = fin 
            total_cost += p.f + p.c * max(0, p.get_makesapn() - p.b)
        order_finished = {o : max([finished_time[p.id][o] for p in processors]) for o in order_perm}
        for o in orders_info:
            total_cost += order_finished[o] * orders_info[o]['weight']
        return total_cost
    
    @staticmethod
    def order_finish_time(processors:dict[str:Processor], order_id:str, order_perm:list[str]) -> int:
        return max([p.getOrderFinishedTimeinP(order_id, order_perm) for p in processors.values()])
    

# def draw_gannt_chart(processors:set[Processor], orders_perm:list[str], image_name:str, title:str, cost:int) -> None:
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
    xticks = np.arange(max([p.get_makesapn() for p in sol.processors.values()]) + 1)
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


def data_init() -> tuple[dict[str:float], dict[str:dict[str:set|float], dict[str:list[float]]]]:
    data_generator = DataGenerator()
    data_generator.set_job_params('normal', job_num=20, job_mu=5, job_sigma=2)
    data_generator.set_order_parm('normal', order_num=5, order_mu=5, order_sigma=2)
    data_generator.set_processo_parm(processor_num=3, speed=[1.25,1.5,1.75], base_time=[25,18,15], fixed_charge=[25,36,45], unit_cost=[2,4,6])
    jobs_load = data_generator.get_jobs()
    orders_info = data_generator.get_order()
    processors_info = data_generator.get_processors()

    # job mapping to order
    jobs_order= {j:o for o, dic in orders_info.items() for j in dic['jobs']} 

    return jobs_load, orders_info, processors_info, jobs_order

'''initilize orders and processors objects'''
def op_init(orders_info:dict[str:dict[str:set, str:int]], processors_info:dict[str:list[float]]) -> tuple[dict[str:Order],dict[str:Processor]]:
    orders = {}
    for k, v in orders_info.items():
        orders[k] = Order(v['jobs'], v['weight'], k)

    processors = {}
    for k, v in processors_info.items():
        processors[k] = Processor(k, *v)

    return orders, processors


if __name__ == '__main__':
    # jobs_load = {'j1' : 8, 'j2' : 5, 'j3' : 7, 'j4' : 8, 'j5' : 4, 'j6' : 6, 'j7' : 6, 'j8' : 5, 'j9':11, 'j10':9, 'j11':4, 'j12':8}
    # orders_info = {
    #     'o1' : {'jobs' : {'j4','j5','j11'}, 
    #             'weight' : 3},
    #     'o2' : {'jobs' : {'j3','j6'}, 
    #             'weight' : 4}, 
    #     'o3' : {'jobs' : {'j1','j7','j8','j10'},
    #             'weight' : 5},
    #     'o4' : {'jobs' : {'j2','j9','j12'},
    #             'weight': 6}
    # }
    # # base_time, unit_cost, fixed_charge, speed
    # processors_info = {
    #     "p1" : [25,15,10,1.6],
    #     "p2" : [15,10,15,1.3],
    #     "p3" : [20,12,14,1.7],
    # }
    # # job mapping to order
    # jobs_order= {j:o for o, dic in orders_info.items() for j in dic['jobs']} 


    # # build order object list
    # orders = {}
    # for k, v in orders_info.items():
    #     orders[k] = Order(v['jobs'], v['weight'], k)

    # processors = {}
    # for k, v in processors_info.items():
    #     processors[k] = Processor(k, *v)


    args = args_parser()
    print(args)
    jobs_load, orders_info, processors_info, jobs_order = get_data(args, DataGenerator)


    orders_perm = ['wml', 'ml', 'wp']
    jobs_perm = ['slf', 'llf']
    saving_prefix = args.saving_folder + '/' + f'{args.job_num}_{args.order_num}_{args.processor_num}'
    pathlib.Path(saving_prefix).mkdir(parents=True, exist_ok=True) 

    for op in orders_perm:
        for jp in jobs_perm:
            orders, processors = op_init(orders_info, processors_info)
            heuristic = Heuristic(orders, processors)
            sol = heuristic.fbs(op, jp)
            draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_fbs.png', '')

            tabulist = TabuList()
            tabusearch = Tabusearch(sol, tabulist, 0.3)
            sol = tabusearch.vnts()
            draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_fbs_tabuSearch.png', '')

            orders, processors = op_init(orders_info, processors_info)
            heuristic = Heuristic(orders, processors)
            sol = heuristic.ls(op, jp)
            draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_ls.png', '')

            tabulist = TabuList()
            tabusearch = Tabusearch(sol, tabulist, 0.3)
            sol = tabusearch.vnts()
            draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_ls_tabuSearch.png', '')

            orders, processors = op_init(orders_info, processors_info)
            heuristic = Heuristic(orders, processors)
            sol = heuristic.ls(op, jp)
            draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_bf.png', '')

            tabulist = TabuList()
            tabusearch = Tabusearch(sol, tabulist, 0.3)
            sol = tabusearch.vnts()
            draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_bf_tabuSearch.png', '')

    # temp code
    print('start ip modeling')
    orders, processors = op_init(orders_info, processors_info)
    jobs = list(jobs_load.values())
    orders = {k:[v['jobs'], v['weight']] for k,v in orders_info.items()}
    processors = processors_info
    order_delivery_ip(jobs, orders, processors, '20jobs_5orders_3processors.png')