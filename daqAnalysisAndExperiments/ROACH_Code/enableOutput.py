#!/usr/bin/env python

import corr, sys, time, logging
import ROACH_SETUP


fpga=[]

if __name__ == '__main__':
	roach = '192.168.40.76'

try:
	lh=corr.log_handlers.DebugLogHandler()
	logger = logging.getLogger(roach)
	logger.addHandler(lh)
	logger.setLevel(10)

	print('Connecting to server %s... '%(roach)),
	fpga = corr.katcp_wrapper.FpgaClient(roach, logger=logger)
	time.sleep(1)

	if fpga.is_connected():
		print('ok\n')
	else:
		print('ERROR connecting to server %s.\n'%(roach))
		exit_fail()
	
	print('------------------------')
	
	print('Enabling output...'),
	sys.stdout.flush()
	fpga.write_int('enable', 1)
	fpga.write_int('atten', 0)
	fpga.write_int('pkt_sim_enable', 1)
	fpga.write_int('rst', 3)
	fpga.write_int('rst', 0)
	fpga.write_int('pkt_sim_enable', 0)
	fpga.write_int('data_select', 1)
	#fpga.write_int('rst',3)
	#fpga.write_int('rst',0)


	print('done')
	exit()
except KeyboardInterrupt:
	exit()
except Exception as e:
	print(e)
	exit()
	


