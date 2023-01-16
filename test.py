import matplotlib.pyplot as plt
'''
plt.ylim(80,100)
x = ['Cifar-100', 'I21K']
y = [85.55, 98.75]
plt.plot(x, y, 'o', label = 'ViT-B')

y = [83.43, 98.95]
plt.plot(x, y, '*', label = 'ViT-L')

y = [83.67, 97.64]
plt.plot(x, y, 'x', label = 'ViT-H')

plt.xlabel('Pretrain model')
plt.ylabel('Accuracy')

plt.legend()
plt.grid()
plt.show()
''' 

import subprocess, os
from multiprocessing import Process

def foo(str):
    os.execl('C:\\Program Files\\DAUM\\PotPlayer\\PotPlayerMini64.exe', 'PotPlayerMini64.exe', str)

  
str = 'F:\\New folder\\1\\ipx-202.mp4'
p = Process(target=foo, args=(str, ))
if __name__ == '__main__':
    p.start()
    print('Yes')