import yaml
import argparse


def args_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()

    parser.add_argument("-c", "--config", type=str, default=None)
    parser.add_argument("--heuristic", type=bool, default=False)
    parser.add_argument("--saving_folder", type=str, default='imgs/heuristic')
    parser.add_argument("--alpha", type=float, default=0.5)

    # job
    parser.add_argument("--job_distribution", type=str, default='normal')
    parser.add_argument("--job_num", type=int, default=20)
    parser.add_argument("--job_mu", type=float, default=5)
    parser.add_argument("--job_sigma", type=float, default=2)
    parser.add_argument("--job_lb", type=float, default=1)
    parser.add_argument("--job_ub", type=float, default=10)
    
    # order
    parser.add_argument("--order_distribution", type=str, default='normal')
    parser.add_argument("--order_num", type=int, default=5)
    parser.add_argument("--order_mu", type=float, default=None)
    parser.add_argument("--order_sigma", type=float, default=None)
    parser.add_argument("--order_lb", type=float, default=None)
    parser.add_argument("--order_ub", type=float, default=None) 

    # processor
    parser.add_argument("--processor_num", type=int, default=3)
    parser.add_argument("--speed", type=float, default=None)
    parser.add_argument("--base_time", type=float, default=None)
    parser.add_argument("--fixed_charge", type=float, default=None)
    parser.add_argument("--unit_cost", type=float, default=None)
    
    args = parser.parse_args()
    if args.config:
        with open(args.config, 'r') as f:
            configs = yaml.safe_load(f)    # load the config file
    
    parser.set_defaults(**configs)

    args = parser.parse_args()

    return args

'''
use DataGenerator to get data
return value:
    jobs_load = {'j1' : 8, ...}
    orders_info  = {'o1' : {'jobs' : {'j4','j5','j11'}, 'weight' : 3}, ...}
    
    # base_time, unit_cost, fixed_charge, speed
    processors_info = { 'p1' : [25,15,10,1.6], ...}
    jobs_order = {'j1' : 'o1', ...}
'''
def get_data(args, DataGenerator) -> tuple[dict[str:float], dict[str:dict[str:set|float], dict[str:list[float]]]]:
    data_generator = DataGenerator()
    data_generator.set_job_params(job_distribution=args.job_distribution, job_num=args.job_num, job_mu=args.job_mu, job_sigma=args.job_sigma)
    data_generator.set_order_parm(order_distribution=args.order_distribution, order_num=args.order_num, order_mu=args.order_mu, order_sigma=args.order_sigma)
    data_generator.set_processo_parm(processor_num=args.processor_num, speed=args.speed, base_time=args.base_time, fixed_charge=args.fixed_charge, unit_cost=args.unit_cost)
    jobs_load = data_generator.get_jobs()
    orders_info = data_generator.get_order()
    processors_info = data_generator.get_processors()

    # job mapping to order
    jobs_order= {j:o for o, dic in orders_info.items() for j in dic['jobs']} 

    return jobs_load, orders_info, processors_info, jobs_order



