from utils.util import args_parser, get_data
from heuristic import Heuristic
import random
import numpy as np
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path

from utils.visualization import draw_gannt_chart
from ip_w_th2 import order_delivery_ip

if __name__ == '__main__':
    
    # parse parameters
    args = args_parser()
    logger.info(f'input parameters : {args}')

    # set random seed
    logger.info(f'set random seed : {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    min_calculate = defaultdict(lambda:0)

    for i in tqdm(range(args.instance_num)):

        # get data
        jobs_load, orders_info, processors_info, jobs_order = get_data(args)
        helper_data = {
            'jobs_load' : jobs_load,
            'jobs_order' : jobs_order,
            'orders_info' : orders_info,
            'processors_info' : processors_info
        }

        orders_perm = ['ml', 'wml', 'wp']
        jobs_perm = ['slf', 'llf']
        draw = True if i == 0 else False
        saving_prefix = args.saving_folder + '/' + f'{args.job_num}_{args.order_num}_{args.processor_num}'
        Path(saving_prefix).mkdir(parents=True, exist_ok=True) 
        res = defaultdict(list) 
        for op in orders_perm:
            for jp in jobs_perm:

                heuristic = Heuristic(helper_data=helper_data, alpha=args.alpha)
                sol = heuristic.fbs(op, jp)
                if draw:
                    draw_gannt_chart(jobs_load, sol, f'{saving_prefix}/{op}_{jp}_fbs.png', '')
                res[(op, jp, 'fbs')].append(sol.getCost())

                # tabulist = TabuList()
                # tabusearch = Tabusearch(sol, tabulist, 0.3)
                # sol = tabusearch.vnts()
                # if draw:
                #     draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_fbs_tabuSearch.png', '')

                heuristic = Heuristic(helper_data=helper_data, alpha=args.alpha)
                sol = heuristic.ls(op, jp)
                if draw:
                    draw_gannt_chart(jobs_load, sol, f'{saving_prefix}/{op}_{jp}_ls.png', '')
                res[(op, jp, 'ls')].append(sol.getCost())

                # tabulist = TabuList()
                # tabusearch = Tabusearch(sol, tabulist, 0.3)
                # sol = tabusearch.vnts()
                # if draw:
                #     draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_ls_tabuSearch.png', '')

                heuristic = Heuristic(helper_data=helper_data, alpha=args.alpha)
                sol = heuristic.bf(op, jp)
                if draw:
                    draw_gannt_chart(jobs_load, sol, f'{saving_prefix}/{op}_{jp}_bf.png', '')
                res[(op, jp, 'bf')].append(sol.getCost())

                # tabulist = TabuList()
                # tabusearch = Tabusearch(sol, tabulist, 0.3)
                # sol = tabusearch.vnts()
                # if draw:
                #     draw_gannt_chart(sol, f'{saving_prefix}/{op}_{jp}_bf_tabuSearch.png', '')

        min_calculate[min(res, key=res.get)] += 100 / args.instance_num
      
    print(min_calculate)
    for k in res:
        print(f'{k} mean csot :', sum(res[k]) / len(res[k]))

    # # temp code
    # # print('start ip modeling')
    # orders, processors = op_init(orders_info, processors_info)
    jobs = list(jobs_load.values())
    orders = {k:[v['jobs'], v['weight']] for k,v in orders_info.items()}
    processors = processors_info

    order_delivery_ip(jobs, orders, processors_info, args.alpha, '20jobs_5orders_3processors.png')