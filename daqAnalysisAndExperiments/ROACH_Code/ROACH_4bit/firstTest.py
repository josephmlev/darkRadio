#from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
#mp.set_start_method('spawn')
import ROACH_utilv2
import time 
import sys 
sys.path.append("../") 
import ROACH_funcsv2
#import functools
import time
class RunObject(object):

    def __init__(self):        
        self.ROACHInst = ROACH_utilv2.ROACH(recompile = False)
    def beginRun(self):
        self.ROACHInst.startRun()


if __name__ == "__main__":
    aROACH = RunObject()
    time.sleep(2)
    print('BEGINNING RUN')
    aROACH.beginRun()
			




