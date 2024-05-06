from utils.util import args_parser, get_data
from heuristic import Heuristic
import random
import numpy as np
import copy
from loguru import logger
from tqdm import tqdm
from pathlib import Path
from time import time 
from datetime import datetime
import csv
import shutil

from utils.visualization import draw_gannt_chart
from ip_w_th2 import order_delivery_ip
from tabu_search import Tabusearch, VNS

if __name__ == '__main__':

    # parse parameters
    args = args_parser()
    logger.info(f'input parameters : {args}')

    # set random seed
    logger.info(f'set random seed : {args.seed}')
    random.seed(args.seed)
    np.random.seed(args.seed)

    ctime_str = datetime.now().strftime("%m-%d_%H-%M-%S")
    root_path = Path('res', ctime_str)
    root_path.mkdir(parents=True, exist_ok=True)
    # backup config file
    shutil.copyfile(args.config, root_path / 'config.yaml')
    
    
    # prepare output file
    if args.record:
        heuristic_fieldnames = ['time', 'obj', 'instance_no','job_perm', 'order_perm', 'heuristic','proceesor_num', 'order_num', 'job_num']
        meta_heuristic_fieldnames = ['time', 'heuristic_obj', 'meta_obj', 'instance_no','job_perm', 'order_perm', 'heuristic','proceesor_num', 'order_num', 'job_num', 'exceeded']
        ip_fieldnames = ['time','obj','optimal','proceesor_num','order_num','job_num']
        if args.heuristic:
            hf = open(root_path / 'heuristic.csv', 'w', newline='')
            heu_writer = csv.DictWriter(hf, fieldnames=heuristic_fieldnames)
            heu_writer.writeheader()

        if args.vns:
            vf = open(root_path / 'vns.csv', 'w', newline='')
            vns_writer = csv.DictWriter(vf, fieldnames=meta_heuristic_fieldnames)
            vns_writer.writeheader()
        
        if args.tabu:
            tf = open(root_path / 'tabu.csv', 'w', newline='')
            tabu_writer = csv.DictWriter(tf, fieldnames=meta_heuristic_fieldnames)
            tabu_writer.writeheader()
        
        if args.ip:
            ipf = open(root_path / 'ip.csv', 'w', newline='')
            ip_writer = csv.DictWriter(ipf, fieldnames=ip_fieldnames)
            ip_writer.writeheader()

    orders_perm = ['ml', 'wml', 'wp']
    jobs_perm = ['slf', 'llf']

    for job_num in args.job_num:
        for order_num in args.order_num:
            for processor_num in args.processor_num:
                logger.info(f'current paramters : job_num {job_num}, order_num {order_num}, processor {processor_num}')
                diff_max = float('-inf')
                for i in tqdm(range(args.instance_num)):
                    # get data
                    jobs_load, orders_info, processors_info, jobs_order = get_data(args, job_num=job_num, order_num=order_num, processor_num=processor_num, beta=args.distribution_beta)
                    data = {
                        'jobs_load' : jobs_load,
                        'jobs_order' : jobs_order,
                        'orders_info' : orders_info,
                        'processors_info' : processors_info
                    }
                    draw = True if i == 0 and args.draw else False
                    if draw:
                        fig_folder =  root_path / f'{job_num}_{order_num}_{processor_num}' 
                        fig_folder.mkdir(parents=True, exist_ok=True)

                    if args.ip and i == 0:
                        start_time = time()
                        is_optimal, obj = order_delivery_ip(jobs_load, orders_info, processors_info, args.alpha, fig_folder / 'ip.png' if draw else '')
                        if args.record:
                            ip_writer.writerow({'time': time() - start_time, 
                                                'obj': obj, 
                                                'optimal': is_optimal, 
                                                'proceesor_num': processor_num, 
                                                'order_num': order_num,
                                                'job_num': job_num})

                    if not args.heuristic:
                        continue                    
                    # record heuristic solution
                    heuristic_result = {}
                    for op in orders_perm:
                        for jp in jobs_perm:
                            for hu in args.heuristic_method:
                                heuristic = Heuristic(helper_data=data, alpha=args.alpha)
                                start_time = time()
                                sol = heuristic.run_heuristic(order_perm_type=op, job_perm_type=jp, mode=hu)
                                if draw: 
                                    draw_gannt_chart(jobs_load, sol, fig_folder / f'{op}_{jp}_{hu}.png', '')
                                heuristic_result[(op,jp,hu)] = sol

                                if args.record:
                                    heu_writer.writerow({'time': time() - start_time, 
                                                        'obj': sol.getCost(), 
                                                        'instance_no': i+1,
                                                        'job_perm': jp, 
                                                        'order_perm': op, 
                                                        'heuristic': hu,
                                                        'proceesor_num': processor_num, 
                                                        'order_num': order_num,
                                                        'job_num': job_num})

                    min_heu = min(heuristic_result, key=lambda x: heuristic_result[x].getCost())
                    max_heu = max(heuristic_result, key=lambda x: heuristic_result[x].getCost())
                    if heuristic_result[max_heu].getCost() - heuristic_result[min_heu].getCost() > diff_max:
                        diff_max = heuristic_result[max_heu].getCost() - heuristic_result[min_heu].getCost()
                        min_heu_disc = min_heu
                        max_heu_disc = max_heu
                        min_heu_disc_sol = heuristic_result[min_heu]
                        max_heu_disc_sol = heuristic_result[max_heu]
                        
                     
                op, jp, hu = min_heu_disc
                heuristic_obj = min_heu_disc_sol.getCost()
                    
                if args.tabu:
                    tabusearch = Tabusearch(copy.deepcopy(min_heu_disc_sol), 0.3)
                    logger.info('start tabu searching...')
                    start_time = time()
                    sol, exceeded = tabusearch.tabu_search()
                    logger.info(f'tabu search finish, total time : {time() - start_time}s')
                    if args.record:
                        tabu_writer.writerow({'time': time() - start_time, 
                                    'heuristic_obj' : heuristic_obj,
                                    'meta_obj': sol.getCost(), 
                                    'instance_no': i+1,
                                    'job_perm': jp, 
                                    'order_perm': op, 
                                    'heuristic': hu,
                                    'proceesor_num': processor_num, 
                                    'order_num': order_num,
                                    'job_num': job_num,
                                    'exceeded': exceeded})
                    if draw:
                        draw_gannt_chart(jobs_load, sol, fig_folder / f'{op}_{jp}_{hu}_tabu.png', '')

                if args.vns:
                    vns = VNS(copy.deepcopy(min_heu_disc_sol), args.neighbor_num)
                    logger.info('start VNS searching...')
                    start_time = time()
                    sol, exceeded = vns.vns()
                    logger.info(f'VNS finish, total time : {time() - start_time}s')
                    if args.record:
                        vns_writer.writerow({'time': time() - start_time, 
                                    'heuristic_obj' : heuristic_obj,
                                    'meta_obj': sol.getCost(), 
                                    'instance_no': i+1,
                                    'job_perm': jp, 
                                    'order_perm': op, 
                                    'heuristic': hu,
                                    'proceesor_num': processor_num, 
                                    'order_num': order_num,
                                    'job_num': job_num,
                                    'exceeded': exceeded})
                    if draw:
                        draw_gannt_chart(jobs_load, sol, fig_folder / f'{op}_{jp}_{hu}_vns.png', '')
                