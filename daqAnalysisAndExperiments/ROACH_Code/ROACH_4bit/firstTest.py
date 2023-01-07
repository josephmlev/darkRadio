#from pathos.multiprocessing import ProcessingPool as Pool
import multiprocessing as mp
#mp.set_start_method('spawn')
import ROACH_utilv2
import time 
import sys 
sys.path.append("../") 
import ROACH_funcsv2
#import functools
#import cupy as cp 
class RunObject(object):

    def __init__(self):        
        self.ROACHInst = ROACH_utilv2.ROACH(recompile = False)
    def beginRun(self):
        self.ROACHInst.startRun()


if __name__ == "__main__":
    #mempool = cp.get_default_memory_pool()
    #pinned_mempool = cp.get_default_pinned_memory_pool()
    #mempool.free_all_blocks()
    #pinned_mempool.free_all_blocks()
    aROACH = RunObject()
    time.sleep(2)
    print('BEGINNING RUN')
    aROACH.beginRun()
			




