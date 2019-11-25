import os

for i in range(10):
    os.system('python train.py -d --iteration ' + str(i))
