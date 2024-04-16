from utils.util import args_parser, get_data
from heuristic import Heuristic
import random
import numpy as np
import pandas as pd
from loguru import logger
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
import time
from datetime import datetime
import csv

from utils.visualization import draw_gannt_chart
from ip_w_th2 import order_delivery_ip
from tabu_search import Tabusearch

if __name__ == '__main__':
    
    # parse parameters
    args = args_parser()
    logger.info(f'input parameters : {args}')

    # set random seed
    logger.info(f'set random seed : {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    # prepare output file
    if args.record:
        ctime_str = datetime.now().strftime("%m-%d_%H-%M")
        heu_path = Path('res', ctime_str)
        heu_path.mkdir(parents=True, exist_ok=True)
        vns_path = Path('res', ctime_str)
        vns_path.mkdir(parents=True, exist_ok=True)

        hf = open(Path(heu_path, 'heuristic.csv'), 'w', newline='')
        vf = open(Path(vns_path, 'vns.csv'), 'w', newline='')

        heu_fieldnames = ['time', 'obj', 'instance_no','job_perm', 'order_perm', 'heuristic','proceesor_num', 'order_num', 'job_num']
        heu_writer = csv.DictWriter(hf, fieldnames=heu_fieldnames)
        heu_writer.writeheader()

        vns_fieldnames = ['time', 'heuristic_obj', 'vns_obj', 'instance_no','job_perm', 'order_perm', 'heuristic','proceesor_num', 'order_num', 'job_num', 'exceeded']
        vns_writer = csv.DictWriter(vf, fieldnames=vns_fieldnames)
        vns_writer.writeheader()


    result = defaultdict(list)

    if args.ip:
        for job_num in args.job_num:
            for order_num in args.order_num:
                for processor_num in args.processor_num:
                    logger.info(f'current paramters : job_num {job_num}, oredr_num {order_num}, processor {processor_num}')
                    jobs_load, orders_info, processors_info, jobs_order = get_data(args, job_num=job_num, order_num=order_num, processor_num=processor_num)
                    start_time = time.time()
                    is_optimal, obj = order_delivery_ip(jobs_load, orders_info, processors_info, args.alpha, f'imgs/ip/{job_num}jobs_{order_num}orders_{processor_num}processors.png')
                    result['time'].append(time.time() - start_time)
                    result['obj'].append(obj)
                    result['optimal'].append(is_optimal)
                    result['proceesor_num'].append(processor_num)
                    result['order_num'].append(order_num)
                    result['job_num'].append(job_num)
    
        if args.record:
            ctime_str = datetime.now().strftime("%m/%d_%H:%M")
            result = pd.DataFrame(result)
            result.to_csv(f'res/ip_result{ctime_str}.csv')
    if not args.heuristic:
        exit()


    orders_perm = ['ml', 'wml', 'wp']
    jobs_perm = ['slf', 'llf']


    result = defaultdict(list)
    tabu_result = defaultdict(list)
      
    for job_num in args.job_num:
        for order_num in args.order_num:
            for processor_num in args.processor_num:
                logger.info(f'current paramters : job_num {job_num}, order_num {order_num}, processor {processor_num}')
                for i in tqdm(range(args.instance_num)):
                    # get data
                    jobs_load, orders_info, processors_info, jobs_order = get_data(args, job_num=job_num, order_num=order_num, processor_num=processor_num)
                    helper_data = {
                        'jobs_load' : jobs_load,
                        'jobs_order' : jobs_order,
                        'orders_info' : orders_info,
                        'processors_info' : processors_info
                    }
                    draw = True if i == 0 and args.draw else False
                    saving_prefix = args.saving_folder + '/' + f'{args.job_num}_{args.order_num}_{args.processor_num}'
                    Path(saving_prefix).mkdir(parents=True, exist_ok=True) 
                    image_name = f'job_num :{job_num}, order_num:{order_num}, processor_num:{processor_num}'

                    # record heuristic solution
                    heuristic_result = {}
                    for op in orders_perm:
                        for jp in jobs_perm:
                            for hu in args.heuristic_method:
                                heuristic = Heuristic(helper_data=helper_data, alpha=args.alpha)
                                start_time = time.time()
                                sol = heuristic.run_heuristic(order_perm_type=op, job_perm_type=jp, mode=hu)
                                if draw:
                                    draw_gannt_chart(jobs_load, sol, f'{saving_prefix}/{op}_{jp}_{hu}.png', '')
                                heuristic_result[(op,jp,hu)] = sol

                                heu_writer.writerow({'time': time.time() - start_time, 
                                                    'obj': sol.getCost(), 
                                                    'instance_no': i+1,
                                                    'job_perm': jp, 
                                                    'order_perm': op, 
                                                    'heuristic': hu,
                                                    'proceesor_num': processor_num, 
                                                    'order_num': order_num,
                                                    'job_num': job_num})

                    chosen_heu = min(heuristic_result, key=lambda x: heuristic_result[x].getCost())
                    op, jp, hu = chosen_heu
                    sol = heuristic_result[chosen_heu]
                    heuristic_obj = sol.getCost()
                    
 
                    tabusearch = Tabusearch(sol, 0.3)
                    start_time = time.time()
                    sol, exceeded = tabusearch.vnts()
                    # sol, exceeded = tabusearch.tabu_search()
                    vns_writer.writerow({'time': time.time() - start_time, 
                                        'heuristic_obj' : heuristic_obj,
                                        'vns_obj': sol.getCost(), 
                                        'instance_no': i+1,
                                        'job_perm': jp, 
                                        'order_perm': op, 
                                        'heuristic': hu,
                                        'proceesor_num': processor_num, 
                                        'order_num': order_num,
                                        'job_num': job_num,
                                        'exceeded': exceeded})

                    if draw:
                        draw_gannt_chart(jobs_load, sol, f'{saving_prefix}/{op}_{jp}_fbs_tabuSearch.png', image_name)

                    # 291.62 
                    # is_optimal, obj = order_delivery_ip(jobs_load, orders_info, processors_info, args.alpha, f'imgs/ip/{job_num}jobs_{order_num}orders_{processor_num}processors.png')
                    # logger.info(f'optimal obj : {obj}')
                 
                                
    # if args.record:
    #     ctime_str = datetime.now().strftime("%m-%d_%H:%M")

    #     result = pd.DataFrame(result)
    #     result.to_csv(f'res/heuristic_result{ctime_str}.csv')

    #     tabu_result = pd.DataFrame(tabu_result)
    #     tabu_result.to_csv(f'res/tabu_result{ctime_str}.csv')

    # else:
    #     logger.debug(pd.DataFrame(result))
    #     logger.debug(pd.DataFrame(tabu_result))