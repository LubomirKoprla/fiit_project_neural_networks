import os

for i in range(1000):
    os.system('python train.py --iteration ' + str(i))
