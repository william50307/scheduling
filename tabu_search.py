from utils.components import Solution
import copy
import random

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
    def __init__(self, current_solution:Solution, alpha:int=None, iters:int=100):
        self.sol = current_solution
        self.alpha = alpha
        self.iters = iters
        self.tabu_list = TabuList()

    # '''swap two jobs'''
    # @staticmethod
    # def swap_job(sol:Solution, pair:tuple[tuple[str,str]]) -> Solution:
    #     (p1,j1), (p2,j2) = pair
    #     new_sol = copy.deepcopy(sol)
    #     new_sol.processors[p1].remove(j1)
    #     new_sol.processors[p1].add(j2)
    #     new_sol.processors[p2].remove(j
    #     new_sol.processors[p2].add(j1)
    #     return new_sol
    
    # '''remove a job from processor and add to another'''
    # @staticmethod
    # def insert_job(sol, pair:tuple[str,str,str]) -> Solution:
    #     j, p1, p2 = pair
    #     new_sol = copy.deepcopy(sol)
    #     new_sol.processors[p1].remove(j)
    #     new_sol.processors[p2].add(j)
    #     return new_sol     

    def get_neighbors(self, sol:Solution, k:float) -> list[Solution]:
        res = []
        thr = random.random() * 100
        if thr <= k:
            # switch order permutation
            random.shuffle(self.sol.orders_perm)

        '''job swap'''
        for pair in sol.get_all_swap_pairs():
            new_sol = sol.swap_job(pair)
            res.append(new_sol)

        '''Insert'''
        for pair in sol.get_all_insert_pairs():
            new_sol = sol.insert_job(pair)
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

        return best_sol
    