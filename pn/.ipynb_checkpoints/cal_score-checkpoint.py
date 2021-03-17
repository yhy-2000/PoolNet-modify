import os
import shutil
def cal_score(id):
    origin_path='results/run-{}-sal-e'.format(id)
    target_path= 'Evaluate-SOD-master/pred/DSS/ECSSD'
    for w in os.listdir(target_path):
        os.remove(os.path.join(target_path,w))
    for w in os.listdir(origin_path):
        e=w.split('_')
        part=e[0]+'.png'
        shutil.copyfile(os.path.join(origin_path,w),os.path.join(target_path,part))
    utilpath='Evaluate-SOD-master'
    command='python Evaluate-SOD-master/main.py --root_dir {}  --save_dir {}'.format(utilpath,utilpath)
    os.system(command)


if __name__=='__main__':
    cal_score(0)