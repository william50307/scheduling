from collections import defaultdict
from functools import reduce
import copy
from utils.components import Order, OrderPerm, Processor, CostCalculator, Solution
from loguru import logger


class Heuristic():
    def __init__(self, helper_data:dict[str:dict], alpha:float=0.5):
        
        self.alpha = alpha
        self.helper_data = helper_data
        self.orders = {}
        for k, v in helper_data['orders_info'].items():
            self.orders[k] = Order(v['jobs'], v['weight'], k, helper_data['jobs_load'])

        self.processors = {}
        for k, v in helper_data['processors_info'].items():
            self.processors[k] = Processor(k, *v, helper_data['jobs_load'], helper_data['jobs_order'])
        
        self.cost_calculator = CostCalculator(helper_data['jobs_load'], helper_data['orders_info'])

    def run_heuristic(self, order_perm_type:str, job_perm_type:str, mode:str) -> Solution:
        if mode == 'ls':
            return self.ls(order_perm_type=order_perm_type, job_perm_type=job_perm_type)
        elif mode == 'fbs':
            return self.fbs(order_perm_type=order_perm_type, job_perm_type=job_perm_type)
        elif mode == 'bf':
            return self.bf(order_perm_type=order_perm_type, job_perm_type=job_perm_type)
        else:
            logger.error(f'heuristic : {mode} is not exist')

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
                    new_sol = cur_sol.swap_job(pair)
                    if new_sol.getCost() < min_cost:
                        min_cost = new_sol.getCost() 
                        can_sol = new_sol
                
                for pair in cur_sol.get_all_insert_pairs():
                    new_sol = cur_sol.insert_job(pair)
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
                    cost = self.cost_calculator.cost(temp_p, order_perm, self.alpha)
                    if cost < min_cost:
                        min_cost = cost
                        can_p = p 
                can_p.add(j)
            return Solution({p.id : p for p in purchased_p}, order_perm, self.cost_calculator, self.alpha)
