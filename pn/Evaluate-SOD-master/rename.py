import os
if __name__=='__main__':
    rootdir='./pred/DSS/ECSSD'
    for w in os.listdir(rootdir):
       e=w.split('_')
       id=int(e[0])
       os.rename(os.path.join(rootdir,w),os.path.join(rootdir,'.'.join([e[0],'png'])))