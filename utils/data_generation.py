import numpy as np
import random

# job load : U[1,10], U[1,50], U[1,100]
# order, order weigh

# data setting
class DataGenerator():
    def set_job_params(self, job_distribution:str, job_num:int, job_mu:float=None, job_sigma:float=None, job_lb:float=None, job_ub:float=None):
        self.job_load_distri = job_distribution
        self.job_num = job_num
        self.job_mu = job_mu
        self.job_sigma = job_sigma
        self.job_lb = job_lb
        self.job_ub = job_ub
    
    def set_order_parm(self, order_distribution:str, order_num:int, order_mu:float=None, order_sigma:float=None, order_lb:float=None, order_ub:float=None):
        self.order_weight_distri = order_distribution
        self.order_num = order_num
        self.order_mu = order_mu
        self.order_sigma = order_sigma
        self.order_lb = order_lb
        self.order_ub = order_ub

    def set_processo_parm(self, processor_num:int, speed:list[float], base_time:list[int], fixed_charge:list[int], unit_cost:list[int]):
        self.processor_num = processor_num
        self.speed = speed
        self.base_time = base_time
        self.fixed_charge = fixed_charge
        self.unit_cost = unit_cost

    def get_jobs(self) -> dict[str:float]:
        loads = np.random.uniform(self.job_lb, self.job_ub, self.job_num) if self.job_load_distri=='uniform' \
                                                                          else np.random.normal(self.job_mu, self.job_sigma, self.job_num)
        self.jobs_load = {'j'+str(i) : loads[i-1] for i in range(1,self.job_num+1)}
        return self.jobs_load
 
    def get_order(self) -> dict[dict[str:set|float]]:
        self.orders_info = {}
        weights = np.random.uniform(self.order_lb, self.order_ub, self.order_num) if self.order_weight_distri == 'uniform' \
                                                                                  else np.random.normal(self.order_mu, self.order_sigma, self.order_num)
        for i in range(1, self.order_num+1):
            self.orders_info['o'+str(i)] = {'jobs' : set(), 'weight' : weights[i-1]}

        for j in self.jobs_load:
            order_id = 'o'+str(random.randint(1, self.order_num))
            self.orders_info[order_id]['jobs'].add(j)

        return self.orders_info
    
    def get_processors(self) -> dict[str:list[int]]:
        self.processors_info = {}
        for i in range(1, self.processor_num+1):
            self.processors_info['p'+str(i)] = [self.base_time[i-1], self.unit_cost[i-1], self.fixed_charge[i-1], self.speed[i-1]]
        
        return self.processors_info
        


if __name__ == '__main__':
    data_generator = DataGenerator()
    data_generator.set_job_params('normal', job_num=20, job_mu=5, job_sigma=2)
    data_generator.set_order_parm('normal', order_num=5, order_mu=5, order_sigma=2)
    data_generator.set_processo_parm(processor_num=3, speed=[1.25,1.5,1.75], base_time=[25,18,15], fixed_charge=[25,36,45], unit_cost=[2,4,6])
    jobs_load = data_generator.get_jobs()
    orders_info = data_generator.get_order()
    processsors_info = data_generator.get_processors()
    print('jobs_load :', jobs_load)
    print('orders_info :', orders_info)
    print('processors_info :', processsors_info)

