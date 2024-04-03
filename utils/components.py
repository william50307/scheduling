from collections import defaultdict
from functools import reduce
from itertools import combinations
import math
import copy
from typing_extensions import Self

class Order():
    def __init__(self, jobs:list[str], weight:int, order_id:str, jobs_load:dict[str:int]):
        self.id = order_id
        self.weight = weight
        self.jobs = jobs
        self.jobs_slf = sorted(jobs, key=lambda x:jobs_load[x])
        self.jobs_llf = sorted(jobs, key=lambda x:jobs_load[x], reverse=True)
        self.jobs_load = jobs_load 
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
        return sum(self.jobs_load[j] for j in self.jobs)

    def __str__(self):
        return self.id
    
class Processor():
    def __init__(self, id:str, 
                       base_time:int, 
                       unit_cost:int, 
                       fixed_charge:int, 
                       speed:float, 
                       jobs_load:dict[str:int], 
                       jobs_order:dict[str:str]):
        self.id = id
        self.b = base_time
        self.c = unit_cost
        self.f = fixed_charge
        self.s = speed 
        self.jobs = defaultdict(set)
        self.jobs_load = jobs_load
        self.jobs_order = jobs_order 

    '''add single job into processor'''
    def add(self, job:str) -> None:
        self.jobs[self.jobs_order[job]].add(job)

    '''remove single job from processor'''
    def remove(self, job:str) -> None:
        self.jobs[self.jobs_order[job]].remove(job)

    def get_all_jobs(self) -> set:
        return set(reduce(lambda x,y:x.union(y), self.jobs.values()))

    '''add all jobs in that order into processor'''
    def add_order(self, order:Order) -> None:
        self.jobs[order.id] = self.jobs[order.id].union(order.jobs)

    '''get cuurent maskspan'''
    def get_makespan(self) -> int:
        return sum(math.ceil(self.jobs_load[j] / self.s) for s in self.jobs.values() for j in s)

    '''the processors total cost'''
    def cost(self) -> int:
        return self.f + self.c * max(0, self.get_makespan() - self.b)
    
    '''get order finished time in this processor'''
    def getOrderFinishedTimeinP(self, order_id:str, order_perm:list[str]) -> int:
        process_time = 0
        for o in order_perm:
            for j in self.jobs[o]:
                process_time += math.ceil(self.jobs_load[j] / self.s)
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
    the orders are ordered in the non-decreasing order of wg/sum(load)
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

class CostCalculator():
    def __init__(self, jobs_load:dict[str:int], orders_info:dict[str:dict[str:set, str:int]]):
        self.jobs_load = jobs_load
        self.orders_info = orders_info

    def cost(self, processors:set[Processor], order_perm:list[str], alpha:float=0.5) -> int:
        finished_time = {}
        acquisition_cost = 0
        for p in processors:
            fin = {}
            prev = 0
            for o in order_perm:
                fin[o] = prev + sum([math.ceil(self.jobs_load[j] / p.s) for j in p.jobs[o]])
                prev = fin[o]
            finished_time[p.id] = fin 
            acquisition_cost += p.f + p.c * max(0, p.get_makespan() - p.b)
        order_cost = 0
        order_finished = {o : max([finished_time[p.id][o] for p in processors]) for o in order_perm}
        for o in self.orders_info:
            order_cost += order_finished[o] * self.orders_info[o]['weight']
        return alpha * acquisition_cost + (1-alpha) * order_cost
    
    def order_finish_time(self, processors:dict[str:Processor], order_id:str, order_perm:list[str]) -> int:
        return max([p.getOrderFinishedTimeinP(order_id, order_perm) for p in processors.values()])
    

class Solution():
    def __init__(self, processors:dict[str:Processor], orders_perm:list[str], cost_calculator:CostCalculator, alpha:float):
        self.processors = processors
        self.orders_perm = orders_perm
        self.cost_calculator = cost_calculator
        self.alpha = alpha

    def getCost(self):
        return self.cost_calculator.cost(set(self.processors.values()), self.orders_perm, self.alpha)

    ''' used on tabu sarch, find all the job pairs which can be swaped'''
    def get_all_swap_pairs(self) -> list[tuple[tuple[str,str]]]:
        res = []
        for p1, p2 in combinations(self.processors.keys(), 2):
            for o in self.orders_perm: # the permutation does not change the result
                for j in self.processors[p1].jobs[o]:
                    for jj in self.processors[p2].jobs[o]:
                        res.append(((p1,j),(p2,jj))) 
        return res
    
    '''swap two jobs'''
    def swap_job(self, pair:tuple[tuple[str,str]]) -> Self: 
        (p1,j1), (p2,j2) = pair
        new_sol = copy.deepcopy(self)
        new_sol.processors[p1].remove(j1)
        new_sol.processors[p1].add(j2)
        new_sol.processors[p2].remove(j2)
        new_sol.processors[p2].add(j1)
        return new_sol
    

    '''
    used on tabu search, find all the job pairs which can be inserted
    return value is a tuple contains three elements : [job name, processor remove from, processor insert into]
    '''
    def get_all_insert_pairs(self) -> list[tuple[str, str, str]]:
        res = []
        for p1, p2 in combinations(self.processors.keys(), 2):
            for o in self.orders_perm:
                for j in self.processors[p1].jobs[o]:
                    res.append((j, p1, p2))
                for j in self.processors[p2].jobs[o]:
                    res.append((j, p2, p1))
        return res


    '''remove a job from processor and add to another'''
    def insert_job(self, pair:tuple[str,str,str]) -> Self:
        j, p1, p2 = pair
        new_sol = copy.deepcopy(self)
        new_sol.processors[p1].remove(j)
        new_sol.processors[p2].add(j)
        return new_sol     

                
