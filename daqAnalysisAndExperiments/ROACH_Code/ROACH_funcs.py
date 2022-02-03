#!/usr/bin/env python

import corr, time, struct, sys, logging, select, socket, numpy, threading
import ROACH_SETUP

def exit_fail():
	print('FAILURE DETECTED. Log entries:\n',lh.printMessages())
	print('Transmit buffer overflow status: %i' %fpga.read_int('tx_over_status_gbe0'))
#    try:
#        fpga.stop()
#    except: pass
#    raise
	exit()

def exit_clean():
	try:
		for f in fpgas: f.stop()
	except: 
		pass
	exit()

def convertIPToNum(addr):
	vals = addr.split('.')
	numberVal = int(vals[0])*(2**24) + int(vals[1])*(2**16) + int(vals[2])*(2**8) + int(vals[3])
	return numberVal

		
def readFIFOPercent(fpga):
	returnVal = fpga.read_uint('fifo_percent')
	return returnVal

def resetFIFO(fpga):
	fpga.write_int('fifo_rst', 1)
	time.sleep(0.01)
	fpga.write_int('fifo_rst', 0)
	returnVal = fpga.read_uint('fifo_percent')
	return returnVal

def progROACH(fpga):
	#Decide where we're going to send the data, and from which addresses:
	
	source_ip_0  = ROACH_SETUP.GBE0_IP
	fabric_port_0 = ROACH_SETUP.GBE0_PORT     
	UDP_port_0 = ROACH_SETUP.GBE0_PORT 
	UDP_IP_0 = ROACH_SETUP.UDP0_IP
	mac_base_transmit_0 = ROACH_SETUP.MAC0_TRANSMIT
	mac_base_0 = ROACH_SETUP.MAC0_BASE
	tx_core_name_0 = ROACH_SETUP.TX0_CORE_NAME

	source_ip_1 = ROACH_SETUP.GBE1_IP 
	fabric_port_1 = ROACH_SETUP.GBE1_PORT  
	UDP_IP_1 = ROACH_SETUP.UDP1_IP
	UDP_port_1 = ROACH_SETUP.GBE1_PORT
	mac_base_transmit_1 = ROACH_SETUP.MAC1_TRANSMIT
	mac_base_1 = ROACH_SETUP.MAC1_BASE

	pkt_period = ROACH_SETUP.PKT_PERIOD #in FPGA clocks (200MHz)
	payload_len = ROACH_SETUP.PAYLOAD_LEN    #in 64bit words
	tx_core_name_1 = ROACH_SETUP.TX1_CORE_NAME

	#brams=['bram_msb','bram_lsb','bram_oob']

	boffile = ROACH_SETUP.DESIGN_FILE
	#if __name__ == '__main__':
	#	from optparse import OptionParser

	#	p = OptionParser()
	#	p.set_usage('tut2.py <ROACH_HOSTNAME_or_IP> [options]')
	#	p.set_description(__doc__)
	#	p.add_option('-p', '--plot', dest='plot', action='store_true',
	#		help='Plot the TX and RX counters. Needs matplotlib/pylab.')  
	#	p.add_option('-a', '--arp', dest='arp', action='store_true',
	#		help='Print the ARP table and other interesting bits.')  
	 
	#	opts, args = p.parse_args(sys.argv[1:])

	#	if args==[]:
	#		print('Please specify a ROACH board. \nExiting.')
	#		exit()
	#	else:
	#roach = IPAddress
	try:
		if fpga.is_connected():
			print('ok\n')
		else:
			print('ERROR connecting to server %s.\n'%(roach))
			exit_fail()
		
		print('------------------------')
		print('Programming FPGA...'),
		sys.stdout.flush()
		fpga.progdev(boffile)
		time.sleep(10)
		print('ok')

		print('------------------------')
		print('Setting ADC registers...')
		sys.stdout.flush()
		adc_addr = [0x0000, 0x0001, 0x0002, 0x0003, 0x0009, 0x000A, 0x000B, 0x000E, 0x000F]
		adc_val = [0x7FFF, 0xB2FF, 0x007F, 0x807F, 0x23FF, 0x007F, 0x807F, 0x00FF, 0x007F]
		#Set 9 to 23FF/33FF to set I/Q channel as input
		
		print('\nCalibrating...')
		corr.katadc.spi_write_register(fpga, 1, 0x0000, 0x7FFF)
		time.sleep(5)
		corr.katadc.spi_write_register(fpga, 1, numpy.ubyte(0x0000), numpy.ushort(0xFFFF))
		time.sleep(5)
		
		for i in range(len(adc_addr)):
			print('Setting ADC register 0x%04X to 0x%04X' % (adc_addr[i], adc_val[i]))
			#corr.katadc.spi_write_register(fpga, 0, adc_addr[i], adc_val[i])
			corr.katadc.spi_write_register(fpga, 1, adc_addr[i], adc_val[i])
			time.sleep(1)
		#corr.katadc.set_interleaved(fpga, 0, 'I')
		corr.katadc.set_interleaved(fpga, 1, 'I')
		#print corr.katadc.get_ambient_temp(fpga, 1)
		print('ok')


		print('Configuring transmitter core...')
		sys.stdout.flush()
	
		fpga.tap_start('tap0',tx_core_name_0, mac_base_0 + 14, convertIPToNum(source_ip_0), fabric_port_0)
		#fpga.tap_start('tap1',tx_core_name_1, mac_base_1 + 14, convertIPToNum(source_ip_1), fabric_port_1)


		fpga.config_10gbe_core(tx_core_name_0, mac_base_0 + 14, convertIPToNum(source_ip_0), fabric_port_0, mac_base_transmit_0 + numpy.asarray([0]*256))
		#fpga.config_10gbe_core(tx_core_name_1, mac_base_1 + 14, convertIPToNum(source_ip_1), fabric_port_1, mac_base_transmit_1 + numpy.asarray([0]*256))

		print('done')
		print('---------------------------')
		print('Setting-up packet source...')
		sys.stdout.flush()
		
		#fpga.write_int('pkt_sim_period',pkt_period)
		#fpga.write_int('pkt_sim_payload_len',1025)
		#fpga.write_int('pkt_sim_on_len', 200)

		#fpga.write_int('afull_wait', ROACH_SETUP.AFULL_CLOCKS)
		#fpga.write_int('eof_per', ROACH_SETUP.FRAMES_PER_SFP)

		print('done')

		print('Setting-up destination addresses...')
		sys.stdout.flush()
		fpga.write_int('dest_ip0', convertIPToNum(UDP_IP_0))
		fpga.write_int('dest_port0', UDP_port_0)
		#fpga.write_int('dest_ip1', convertIPToNum(UDP_IP_1))
		#fpga.write_int('dest_port1', UDP_port_1)
		fpga.write_int('atten', 0)
		print('done')

		print('Resetting cores and counters...')
		sys.stdout.flush()
		fpga.write_int('rst',3)
		fpga.write_int('rst',0)
		print('done')

		time.sleep(2)

		#if opts.arp:
		print('\n\n===============================')
		print('10GbE Transmitter 0 core details:')
		print('===============================')
		print("Note that for some IP address values, only the lower 8 bits are valid!")
		fpga.print_10gbe_core_details(tx_core_name_0, arp=True)
		#print('\n\n============================')
		#print('10GbE Transmitter 1 core details:')
		#print('============================')
		#print("Note that for some IP address values, only the lower 8 bits are valid!")
		#fpga.print_10gbe_core_details(tx_core_name_1, arp=True)

		#print 'Sent %i packets already from core 0.'%fpga.read_int('gbe0_tx_cnt')
		#print 'Sent %i packets already from core 1.'%fpga.read_int('gbe1_tx_cnt')


		#print '------------------------'
		
		print('Enabling output...')
		sys.stdout.flush()
		fpga.write_int('enable', 0)
		time.sleep(0.1)
		fpga.write_int('pkt_sim_enable', 0)
		time.sleep(0.1)
		fpga.write_int('data_select', 1)
		time.sleep(0.1)
		#fpga.write_int('rst',3)
		#fpga.write_int('rst',0)
		fpga.write_int('fifo_rst', 1)
		time.sleep(0.1)
		fpga.write_int('fifo_rst', 0)
		time.sleep(0.1)

		print('ESTIMATED FPGA CLOCK SPEED: ' + str(fpga.est_brd_clk()))
		print('ATENUATION: ' + str(fpga.read_int('atten')))
		print('done')

		#print 'TRYING SOMETHING...'
		#fpga.snapshot_get('tx_snapshot_in_data', man_trig = False)
		#fpga.write('ctrl', 3)
		#print('FUCK!!!')
		
		#sys.exit(1)
		print('WE MADE IT HERE\n\n\n\n\n')
		sys.stdout.flush()
		print('CURRENT THREADS: ' + str(threading.enumerate()))
		#fpga.stop()
		return
		sys.exit(1)
		#exit_clean()
	except KeyboardInterrupt:
		exit_clean()
	except Exception as e:
		print('THINGS WENT WRONG')
		print('THE ERROR IS: ' + str(e))
		exit_fail()


def enableROACH(fpga):
	#fpga=[]

	#if __name__ == '__main__':
	#roach = IPAddress

	try:
		#lh=corr.log_handlers.DebugLogHandler()
		#logger = logging.getLogger(roach)
		#logger.addHandler(lh)
		#logger.setLevel(10)

		#print('Connecting to server %s... '%(roach)),
		#fpga = corr.katcp_wrapper.FpgaClient(roach, logger=logger)
		#time.sleep(1)
		if fpga.is_connected():
			print('ok\n')
		else:
			print('ERROR connecting to server %s.\n'%(roach))
			exit_fail()
		
		print('------------------------')
		
		print('Core details...')
		fpga.print_10gbe_core_details('gbe0', arp=True)
		print('Enabling output...'),
		sys.stdout.flush()
		#fpga.write_int('dest_ip0', 3232250881)
		fpga.write_int('data_select', 1)
		time.sleep(0.1)
		fpga.write_int('pkt_sim_payload_len',1022)
		time.sleep(0.1)		
		#fpga.write_int('pkt_sim_on_len', 1025)
		#time.sleep(0.1)
		#fpga.write_int('pkt_sim_trig_set', 1023)
		#time.sleep(0.1)
		fpga.write_int('enable', 1)
		time.sleep(0.1)
		fpga.write_int('atten', 0)
		time.sleep(0.1)
		fpga.write_int('pkt_sim_enable', 1)
		time.sleep(0.1)
#		fpga.write_int('rst', 3)
#		time.sleep(0.1)		
#		fpga.write_int('rst', 0)
#		time.sleep(0.1)		
		#time.sleep(10)
		print('PERCENTAGE FULL: ' + str(fpga.read_uint('fifo_percent')>>24))

		print('done')
		#fpga.stop()
		print('TOTAL ACTIVE THREADS: ' + str(threading.active_count()))
		return
		#for thread in threading.enumerate():    
		#	print(type(thread))
		#	if type(thread) != threading._MainThread: #never kill main thread directly
		#		thread._stop()	
		#return
		#exit()
	except KeyboardInterrupt:
		exit()
	except Exception as e:
		print(e)
		exit()
	
	return



