import os
def get_labels():
    root='./data/ECSSD/Imgs'
    res={}
    for w in os.listdir(root):
        e=w.split('.')
        if len(e)==2 and e[1]=='png':
            id=int(e[0])
            res[id]=os.path.join(root,w)
    return res
def get_predicted(num=10):
    root='./results/run-{}-sal-e'.format(int(num))
    res={}
    for w in os.listdir(root):
        e=w.split('.')
        if len(e)==2 and e[1]=='png':
            f=e[0].split('_')
            id=int(f[0])
            res[id]=os.path.join(root,w)
    return res
import cv2
def mae(p1,p2):
    return abs(p1-p2).mean()/255
import numpy as np
if __name__=='__main__':
    d1=get_labels()
    d2=get_predicted(1)
    li=[]
    for w in d1.keys():
        p1=cv2.imread(d1[w])
        p2=cv2.imread(d2[w])
        score=mae(p1,p2)
        li.append(score)
    print(np.array(li).mean())