from utils.components import Solution
import copy
import random
from time import time

class TabuList():
    def __init__(self, tabu_len=7):
        self.tabu_list = []
        self.tabu_len = tabu_len

    def add(self, new_sol):
        if len(self.tabu_list) == self.tabu_len:
            self.tabu_list.pop(0)
            self.tabu_list.append(new_sol.rep)
    
    def __contains__(self, item):
        return item.rep in self.tabu_list

class Tabusearch():
    def __init__(self, current_solution:Solution, alpha:int=None, iters:int=100):
        self.sol = current_solution
        self.alpha = alpha
        self.iters = iters
        self.tabu_list = TabuList()

    def get_tabu_neighbors(self, sol:Solution, k:float):

        thr = random.random() * 100
        if thr <= k:
            # switch order permutation
            random.shuffle(self.sol.orders_perm)

        '''job swap'''
        for pair in sol.get_all_swap_pairs():
            yield sol.swap_job(pair)

        '''Insert'''
        for pair in sol.get_all_insert_pairs():
            yield sol.insert_job(pair)       

    def tabu_search(self) -> Solution:
        best_sol = self.sol
        current_sol = self.sol
        for k in range(self.iters):
            start_time = time()

            if time() - start_time >= 1800:
                return best_sol, True
       
            min_cost = float('inf')
            best_neighbor = None
            for sol in self.get_tabu_neighbors(current_sol, k):
                c = sol.getCost()
                if  c < min_cost and sol not in self.tabu_list:
                    best_neighbor = sol
                    min_cost = c

            if best_neighbor is None:
                print('No non-tabu neighbors found')
                return best_sol
            
            current_sol = best_neighbor
            self.tabu_list.add(best_sol)
    
            if best_neighbor.getCost() < best_sol.getCost():
                best_sol = best_neighbor

        return best_sol, False

class VNS():
    def __init__(self, solution:Solution):
        self.sol = solution

    def local_search(self, sol:Solution, k=int) -> list[Solution]:
        min_cost = float('inf')
        best_neighbor = None
        '''ref: partition problem with VNS'''
        '''Insert'''
        if k % 3 == 1:
            visit_neighbor = sol.variable_neighbor_1()
        elif k % 3 == 2:
            visit_neighbor = sol.variable_neighbor_2()
        else:
            visit_neighbor = sol.variable_neighbor_3()

        # TODO:should return if time exceed...
        for pairs in visit_neighbor:
            sol.insert_inplace(pairs)
            cost = sol.getCost()
            if cost < min_cost:
                min_cost = cost
                best_neighbor = copy.deepcopy(sol)
            sol.insert_resume_inplace(pairs)

        return best_neighbor
    
    def vns(self) -> tuple[Solution, bool]:
        best_sol = self.sol
        current_sol = self.sol
        k_max = len(self.sol.orders_perm) // 2 + 1
        k = 0
        l = 1
        r = 0 
        start_time = time()
        while k <= k_max:
            if time() - start_time >= 1800:
                return best_sol, True

            rand_sol = current_sol.swap2order(k, l)
            best_neighbor = self.local_search(rand_sol, r)
            
            r = (r+1) % 3

            if best_neighbor.getCost() < best_sol.getCost():
                best_sol = best_neighbor
                current_sol = best_neighbor
                k = 0
                l = 1
            elif l == k_max - 1:
                k += 1
                l = 1
            else:
                l += 1        
        return best_sol, False