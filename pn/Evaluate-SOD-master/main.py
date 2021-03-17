import torch
import torch.nn as nn
import argparse
import os.path as osp
import os
from evaluator import Eval_thread
from dataloader import EvalDataset
# from concurrent.futures import ThreadPoolExecutor
def get_last_runid():
    rtdir='results'
    res=0
    for w in os.listdir(rtdir):
        e=w.split('-')
        if len(e)>=2:
            id=int(e[1])
            res=max(res,id)
    return res
def count_txt():
    res=0
    rt='results/run-{}/models'.format(get_last_runid())
    for w in os.listdir(rt):
        e=w.split('.')
        if len(e)==2 and e[1]=='txt':
            res+=1
    return res
def main(cfg):
    root_dir = cfg.root_dir
    if cfg.save_dir is not None:
        output_dir = cfg.save_dir
    else:
        output_dir = root_dir
    gt_dir = osp.join(root_dir, 'gt')
    pred_dir = osp.join(root_dir, 'pred')
    if cfg.methods is None:
        method_names = os.listdir(pred_dir)
    else:
        method_names = cfg.methods.split(' ')
    if cfg.datasets is None:
        dataset_names = os.listdir(gt_dir)
    else:
        dataset_names = cfg.datasets.split(' ')
    
    threads = []
    for dataset in dataset_names:
        for method in method_names:
            loader = EvalDataset(osp.join(pred_dir, method, dataset), osp.join(gt_dir, dataset))
            thread = Eval_thread(loader, method, dataset, output_dir, cfg.cuda)
            threads.append(thread)
    for thread in threads:
        res=thread.run()
        print(res)
        id=get_last_runid()
        cnt_txt=count_txt()
        rt='results/run-{}/models/{}.txt'.format(id,cnt_txt)
        with open(rt,'w')as f:
            print(res,file=f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--methods', type=str, default=None)
    parser.add_argument('--datasets', type=str, default=None)
    parser.add_argument('--root_dir', type=str, default='./')
    parser.add_argument('--save_dir', type=str, default=None)
    parser.add_argument('--cuda', type=bool, default=True)
    config = parser.parse_args()
    main(config)
