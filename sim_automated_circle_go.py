from subprocess import Popen, PIPE, STDOUT
import shlex
import time
import sys
#import psutil
import os, signal 

cnt = 0
for i in range(0,1):
    p1 = Popen(["python3.6","sim_automated_circle.py", str(i)], stdout=PIPE, stderr=PIPE)

    time.sleep(5)
    print("Number of times habitat simulator started: ", cnt)
    cnt+=1

    # for line in out.decode("utf-8").split('\\n'):
    #     print('\t' + line)
    # print('ERROR')
    # for line in err.decode("utf-8").split('\\n'):
    #     print('\t' + line)
    
    p1.terminate()

    # Iterate over all running process

    print("Done with single iteration. Terminating everything")
    print("==========================================================")
