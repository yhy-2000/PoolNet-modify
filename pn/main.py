import argparse
import os
from dataset.dataset import get_loader
from solver import Solver
from torchvision import models
from torchsnooper import snoop
from test_model import test_model
def get_test_info(sal_mode='e'):
    image_root,image_source='',''
    if sal_mode == 'e':
        image_root = './data/ECSSD/Imgs/'
        image_source = './data/ECSSD/test.lst'
    elif sal_mode == 'p':
        image_root = './data/PASCALS/Imgs/'
        image_source = './data/PASCALS/test.lst'
    elif sal_mode == 'd':
        image_root = './data/DUTOMRON/Imgs/'
        image_source = './data/DUTOMRON/test.lst'
    elif sal_mode == 'h':
        image_root = './data/HKU-IS/Imgs/'
        image_source = './data/HKU-IS/test.lst'
    elif sal_mode == 's':
        image_root = './data/SOD/Imgs/'
        image_source = './data/SOD/test.lst'
    elif sal_mode == 't':
        image_root = './data/DUTS-TE/Imgs/'
        image_source = './data/DUTS-TE/test.lst'
    elif sal_mode == 'm_r': # for speed test
        image_root = './data/MSRA/Imgs_resized/'
        image_source = './data/MSRA/test_resized.lst'

    return image_root, image_source

def main(config,test_size_distribuion=False):
    if config.mode == 'train':
        train_loader = get_loader(config)
        run = 0
        if config.resume:
            run=get_last_runid()
        else:
            while os.path.exists("%s/run-%d" % (config.save_folder, run)):
                run += 1
            os.mkdir("%s/run-%d" % (config.save_folder, run))
            os.mkdir("%s/run-%d/models" % (config.save_folder, run))
        config.save_folder = "%s/run-%d" % (config.save_folder, run)
        train = Solver(train_loader, None, config)
        if not test_size_distribuion:
            train.train()
        else:
            train.tr()
    elif config.mode == 'test':
        config.test_root, config.test_list = get_test_info(config.sal_mode)
        test_loader = get_loader(config, mode='test')
        if not os.path.exists(config.test_fold): os.mkdir(config.test_fold)
        test = Solver(None, test_loader, config)
        test.test()
    else:
        raise IOError("illegal input!!!")
        
def get_last_runid():
    rtdir='results'
    res=0
    for w in os.listdir(rtdir):
        e=w.split('-')
        if len(e)>=2:
            id=int(e[1])
            res=max(res,id)
    return res
def get_latest_model_version(runid=get_last_runid()):
    dir='results/run-{}/models'.format(runid)
    resid=-1
    res=''
    for w in os.listdir(dir):
        e=w.split('.')
        if len(e)==2 and e[1]=='pth':
            f=e[0].split('_')
            if len(f)==2:
                curid=int(f[1])
                if curid>resid:
                    resid=curid
                    res=w
    return os.path.join(dir,res)
def get_latest_model_epoch(runid=get_last_runid()):
    dir = 'results/run-{}/models'.format(runid)
    resid = -1
    res = ''
    for w in os.listdir(dir):
        e = w.split('.')
        if len(e) == 2 and e[1] == 'pth':
            f = e[0].split('_')
            if len(f) == 2:
                curid = int(f[1])
                if curid > resid:
                    resid = curid
                    res = w
    return resid
def rank(model_name):
    e=model_name.split('.')
    pre=e[0]
    u=pre.split('_')
    if len(u)==1:
        #final.pth
        return 10000
    return int(u[1])
def get_all_model_name(runid=get_last_runid()):
    rtdir='results/run-{}/models'.format(runid)
    ans=[]
    for w in os.listdir(rtdir):
        e=w.split('.')
        if len(e)==2 and e[1]=='pth':
            ans.append(e[0])
    ans.sort(key=rank,reverse=True)
    return ans

def test_all_model(runid=get_last_runid()):
    for model_name in get_all_model_name(runid):
        test_model(runid,model_name=model_name)

        
def add_parser():
    vgg_path = './dataset/pretrained/vgg16_20M.pth'
    resnet_path = './dataset/pretrained/resnet50_caffe.pth'
    os.environ["CUDA_VISIBLE_DEVICES"] = '1'
    parser = argparse.ArgumentParser()

    # Hyper-parameters
    parser.add_argument('--n_color', type=int, default=3)
<<<<<<< HEAD
    parser.add_argument('--lr', type=float, default=1e-1)  # Learning rate resnet:5e-5, vgg:1e-4
=======
    parser.add_argument('--lr', type=float, default=1e-3)  # Learning rate resnet:5e-5, vgg:1e-4
>>>>>>> 6e03716a207ae53a533559423d1eb0b334fd6cae
    parser.add_argument('--wd', type=float, default=0.0005)  # Weight decay
    parser.add_argument('--no-cuda', dest='cuda', action='store_false')

    # Training settings
    parser.add_argument('--arch', type=str, default='resnet')  # resnet or vgg
    parser.add_argument('--pretrained_model', type=str, default=resnet_path)
    parser.add_argument('--epoch', type=int, default=24)
<<<<<<< HEAD
    parser.add_argument('--batch_size', type=int, default=6)  # only support 1 now
=======
    parser.add_argument('--batch_size', type=int, default=1)  # only support 1 now
>>>>>>> 6e03716a207ae53a533559423d1eb0b334fd6cae
    parser.add_argument('--num_thread', type=int, default=1)
    parser.add_argument('--load', type=str, default='')
    parser.add_argument('--save_folder', type=str, default='./results')
    parser.add_argument('--epoch_save', type=int, default=1)
    parser.add_argument('--iter_size', type=int, default=1)
    parser.add_argument('--show_every', type=int, default=50)
<<<<<<< HEAD
    parser.add_argument('--reduction',type=str,default='mean')
=======
    parser.add_argument('--reduction',type=str,default='sum')
>>>>>>> 6e03716a207ae53a533559423d1eb0b334fd6cae
    parser.add_argument('--show_grad',type=bool,default=True)
    parser.add_argument('--resume',type=bool,default=True)
    parser.add_argument('--test_function',type=str,default='sigmoid')
    parser.add_argument('--optimizer',type=str,default='SGD')

    # Train data
    parser.add_argument('--train_root', type=str, default='./data/DUTS/DUTS-TR')
    parser.add_argument('--train_list', type=str, default='./data/DUTS/DUTS-TR/train_pair.lst')

    # Testing settings
    parser.add_argument('--model', type=str, default=None)  # Snapshot
    parser.add_argument('--test_fold', type=str, default=None)  # Test results saving folder
    parser.add_argument('--sal_mode', type=str, default='e')  # Test image dataset

    # Misc
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'test'])
    config = parser.parse_args()
    config.cuda = True

    if not os.path.exists(config.save_folder):
        os.mkdir(config.save_folder)
    
    return config

    
    
def train(resume=False):
    config=add_parser()
    config.resume=resume
    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list
    main(config)

if __name__ == '__main__':
    config=add_parser()
    
    # Get test set info
    test_root, test_list = get_test_info(config.sal_mode)
    config.test_root = test_root
    config.test_list = test_list

    main(config)