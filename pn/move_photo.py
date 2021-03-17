import shutil
import os

# path="./data/ECSSD/Imgs"
# tar="./Evaluate-SOD-master/gt/ECSSD"
# for p,d,filenames in os.walk(tar):
#     for file in filenames:
#         if file.endswith(".jpg"):
#             ori_path= path+"/"+file
#             tar_path= tar+"/"+file
#             shutil.move(tar_path,ori_path)


import shutil
import os
import cv2

path="./data/ECSSD/Imgs"
tar="./Evaluate-SOD-master/gt/ECSSD"
for p,d,filenames in os.walk(tar):
    for file in filenames:
        if file.endswith(".png"):
            img_path=tar+"/"+file
            img=cv2.imread(img_path)
            img=cv2.resize(img,(300,400))
            cv2.imwrite(img_path,img)