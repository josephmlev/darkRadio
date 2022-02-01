"""
Setup and unique functionality for the narrow-band correlator modes. Here narrowband consists of two filterbanks, with the first one doing a coarse channelisation and the second further channelising one of those channels. As used by KAT-7's narrowband mode.
"""
"""
Revisions:
2011-07-07  PVP  Initial revision.
"""
import numpy, struct, construct, corr_functions, snap

def bin2fp(bits, m = 8, e = 7):
    if m > 32:
        raise RuntimeError('Unsupported fixed format: %i.%i' % (m,e))
    shift = 32 - m
    bits = bits << shift
    m = m + shift
    e = e + shift
    return float(numpy.int32(bits)) / (2**e)

# f-engine adc control
register_fengine_adc_control = 'adc_ctrl0' / construct.BitStruct('enable' / construct.Flag,       # 31    Enable input channel on KAT ADC.
    construct.Padding(32 - 6 - 1),  # 6-30
    'atten' / construct.BitsInteger(6)) # 0-5   KAT ADC channel attenuation setting.

# f-engine status
register_fengine_fstatus = 'fstatus0' / construct.BitStruct('coarse_bits' / construct.BitsInteger(5),       # 27-31 2^x - the number of points in the coarse FFT.
    'fine_bits' / construct.BitsInteger(5),         # 22-26 2^y - the number of points in the fine FFT.
    'sync_val' / construct.BitsInteger(2),          # 20-21 On which ADC cycle did the sync happen?
    construct.Padding(2),                       # 18-19
    'xaui_lnkdn' / construct.Flag,               # 17    The 10GBE link is down.
    'xaui_over' / construct.Flag,                # 16    The 10GBE link has overflows.
    construct.Padding(9),                       # 7-15
    'clk_err' / construct.Flag,                  # 6     The board frequency is calculated out of bounds.
    'adc_disabled' / construct.Flag,             # 5     The ADC has been disabled.
    'ct_error' / construct.Flag,                 # 4     There is a QDR error from the corner-turner.
    'adc_overrange' / construct.Flag,            # 3     The ADC is reporting over-ranging.
    'fine_fft_overrange' / construct.Flag,       # 2     Not used currently.
    'coarse_fft_overrange' / construct.Flag,     # 1     The coarse FFT is over-range.
    'quant_overrange' / construct.Flag)          # 0     The quantiser is over-range.

# f-engine coarse control
register_fengine_coarse_control = 'coarse_ctrl' / construct.BitStruct(construct.Padding(32 - 10 - 10 - 1 - 1 - 6),    # 28-31
    'debug_chan' / construct.BitsInteger(6),            # 22-27 Which channel to capture when debugging.
    'debug_specify_chan' / construct.Flag,           # 21    Capture only a specific channel when debugging.
    'debug_pol_select' / construct.Flag,             # 20    Select a polarisation for debugging.
    'channel_select' / construct.BitsInteger(10),       # 10-19 Which channel should be fine channelised?
    'fft_shift' / construct.BitsInteger(10))            # 0-9   The shift-schedule for the coarse FFT.

# f-engine fine control
register_fengine_fine_control = 'fine_ctrl' / construct.BitStruct(construct.Padding(32 - 26),             	# 26-31
    'fft_shift' / construct.BitsInteger(26))        # 0-25  Fine FFT shift schedule - not currently used.

# f-engine control
register_fengine_control = 'control' / construct.BitStruct(construct.Padding(4),                       # 28-31
    'debug_snap_select' / construct.BitsInteger(3), # 25-27 Select the source to route to the general debug snap block.
    construct.Padding(1),                       # 24
    construct.Padding(2),                       # 22-23
    'tvgsel_fine' / construct.Flag,              # 21    Fine channelisation TVG enable.
    'tvgsel_adc' / construct.Flag,               # 20    ADC TVG enable.
    'tvgsel_fdfs' / construct.Flag,              # 19    Fine-delay and fringe-stopping TVG enable.
    'tvgsel_pkt' / construct.Flag,               # 18    Packetiser TVG enable.
    'tvgsel_ct' / construct.Flag,                # 17    Corner-turner TVG enable.
    'tvg_en' / construct.Flag,                   # 16    Global TVG enable.
    construct.Padding(4),                       # 12-15
    'flasher_en' / construct.Flag,               # 11    Enable the "knightrider" pattern on the front LED panel.
    'adc_protect_disable' / construct.Flag,      # 10    Disable the protection on the ADC front-end.
    'gbe_enable' / construct.Flag,               # 9     Enable the 10GBE core.
    'gbe_rst' / construct.Flag,                  # 8     Reset the 10GBE core.
    construct.Padding(4),                       # 4-7
    'clr_status' / construct.Flag,               # 3     Clear the status registers.
    'arm' / construct.Flag,                      # 2     Arm the board.
    'man_sync' / construct.Flag,                 # 1     Force a board sync.
    'sys_rst' / construct.Flag)                  # 0     Reset the board.

# x-engine control
register_xengine_control = 'ctrl' / construct.BitStruct(construct.Padding(32 - 16 - 1),     # 17-31
    'gbe_out_enable' / construct.Flag,   # 16    Enable the 10GBE core.
    'gbe_rst' / construct.Flag,          # 15    Reset the 10GBE core.
    construct.Padding(15 - 12 - 1),     # 13-14
    'flasher_en' / construct.Flag,       # 12    Enable the "knightrider" pattern on the front LED panel.
    'gbe_out_rst' / construct.Flag,      # 11
    'loopback_mux_rst' / construct.Flag, # 10    
    'gbe_enable' / construct.Flag,       # 9     Enable the 10GBE core.
    'cnt_rst' / construct.Flag,          # 8     Reset the packet counter.
    'clr_status' / construct.Flag,       # 7     Clear the status registers.
    construct.Padding(7 - 0 - 1),       # 1-6
    'vacc_rst' / construct.Flag)         # 0     Reset the vector accumulator.

# x-engine status
register_xengine_status = 'xstatus0' / construct.BitStruct(construct.Padding(32 - 18 - 1),     # 19-31
    'gbe_lnkdn' / construct.Flag,        # 18
    'xeng_err' / construct.Flag,         # 17
    construct.Padding(17 - 5 - 1),      # 6-16
    'vacc_err' / construct.Flag,         # 5
    'rx_bad_pkt' / construct.Flag,       # 4
    'rx_bad_frame' / construct.Flag,     # 3
    'tx_over' / construct.Flag,          # 2
    'pkt_reord_err' / construct.Flag,    # 1
    'pack_err' / construct.Flag)         # 0

# x-engine tvg control
register_xengine_tvg_sel = 'tvg_sel' / construct.BitStruct(construct.Padding(32 - 1 - 2 - 2 - 6),  # 11-31
    'vacc_tvg_sel' / construct.BitsInteger(6),  # 5-10
    'xeng_tvg_sel' / construct.BitsInteger(2),  # 3-4
    'descr_tvg_sel' / construct.BitsInteger(2), # 1-2
    'xaui_tvg_sel' / construct.Flag)         # 0

# the snap_rx block on the x-engine
snap_xengine_rx = 'snap_rx0' / construct.BitStruct(construct.Padding(128 - 64 - 16 - 5 - 28 - 15), 
    'ant' / construct.BitsInteger(15), 
    'mcnt' / construct.BitsInteger(28), 
    'loop_ack' / construct.Flag,
    'gbe_ack' / construct.Flag,
    'valid' / construct.Flag,
    'eof' / construct.Flag,
    'flag' / construct.Flag,
    'ip_addr' / construct.BitsInteger(16),
    'data' / construct.BitsInteger(64))

# the raw gbe rx snap block on the x-engine
snap_xengine_gbe_rx = 'snap_gbe_rx0' / construct.BitStruct(construct.Padding(128 - 64 - 32 - 7),
    'led_up' / construct.Flag,
    'led_rx' / construct.Flag,
    'eof' / construct.Flag,
    'bad_frame' / construct.Flag,
    'overflow' / construct.Flag,
    'valid' / construct.Flag,
    'ack' / construct.Flag,
    'ip_addr' / construct.BitsInteger(32),
    'data' / construct.BitsInteger(64))

# the snap block immediately after the x-engine
snap_xengine_vacc = 'snap_vacc0' / construct.BitStruct('data' / construct.BitsInteger(32))

def fft_shift_coarse_set_all(correlator, shift = -1):
    """
    Set the per-stage shift for the coarse channelisation FFT on all correlator f-engines.
    """    
    if shift < 0:
        shift = correlator.config['fft_shift_coarse']
    corr_functions.write_masked_register(correlator.ffpgas, register_fengine_coarse_control, fft_shift = shift)
    correlator.syslogger.info('Set coarse FFT shift patterns on all F-engines to 0x%x.' % shift)

def fft_shift_fine_set_all(correlator, shift = -1):
    """
    Set the per-stage shift for the fine channelisation FFT on all correlator f-engines.
    """
    if shift < 0:
        shift = correlator.config['fft_shift_fine']
    corr_functions.write_masked_register(correlator.ffpgas, register_fengine_fine_control, fft_shift = shift)
    correlator.syslogger.info('Set fine FFT shift patterns on all F-engines to 0x%x.' % shift)

def fft_shift_get_all(correlator):
    """
    Get the current FFT shift settings, coarse and fine, for all correlator f-engines.
    """
    rv = {}
    for in_n, ant_str in enumerate(correlator.config._get_ant_mapping_list()):
        ffpga_n, xfpga_n, fxaui_n, xxaui_n, feng_input = correlator.get_ant_str_location(ant_str)
        coarse_ctrl = corr_functions.read_masked_register([correlator.ffpgas[ffpga_n]], register_fengine_coarse_control)
        fine_ctrl = corr_functions.read_masked_register([correlator.ffpgas[ffpga_n]], register_fengine_fine_control)
        rv[ant_str] = [coarse_ctrl[0]['fft_shift'], fine_ctrl[0]['fft_shift']]
    return rv

def feng_status_get(c, ant_str):
    """
    Reads and decodes the status register for a given antenna. Adds some other bits 'n pieces relating to Fengine status.
    """
    ffpga_n, xfpga_n, fxaui_n, xxaui_n, feng_input = c.get_ant_str_location(ant_str)
    rv = corr_functions.read_masked_register([c.ffpgas[ffpga_n]], register_fengine_fstatus, names = ['fstatus%i' % feng_input])[0]
    if rv['xaui_lnkdn'] or rv['xaui_over'] or rv['clk_err'] or rv['ct_error'] or rv['fine_fft_overrange'] or rv['coarse_fft_overrange']:
        rv['lru_state']='fail'
    elif rv['adc_overrange']:
        rv['lru_state']='warning'
    else:
        rv['lru_state']='ok'
    return rv

def channel_select(c, freq_hz = -1, specific_chan = -1, selectchan = True):
    """
    Set the coarse channel based on a given center frequency, given in Hz.
    """
    if not c.is_narrowband():
        raise RuntimeError('This command cannot be run in the current mode.')
    if freq_hz != -1 and specific_chan != -1:
        raise RuntimeError('Specify a frequency in Hz OR a specific coarse channel, not both.')
    elif freq_hz == -1 and specific_chan == -1:
        raise RuntimeError('Specify frequency in Hz or specific coarse channel.')
    channel_bw = c.config['bandwidth']
    coarse_chans = c.config['coarse_chans']
    if specific_chan != -1:
        chan = specific_chan
        freq_hz = chan * channel_bw
    else:
        total_bw = c.config['rf_bandwidth']
        chan = int(round(freq_hz / total_bw * coarse_chans))
    if chan >= coarse_chans:
        raise RuntimeError('Coarse channel too large: %i >= %i' % (chan, coarse_chans))
    chan_cf = chan * channel_bw
    if selectchan:
        try:
            corr_functions.write_masked_register(c.ffpgas, register_fengine_coarse_control, channel_select = chan)
            c.config['center_freq'] = chan_cf
            c.config['current_coarse_chan'] = chan
            c.spead_narrowband_issue()
        except:
            errmsg = 'Something bad happened trying to write the coarse channel select register.'
            c.syslogger.error(errmsg)
            raise RuntimeError(errmsg)
    return chan_cf, chan, freq_hz - chan_cf

"""
SNAP blocks in the narrowband system.
"""

snap_adc = 'adc_snap'
snap_debug = 'snap_debug'

snap_fengine_adc = snap_adc / construct.BitStruct('d0_0' / construct.BitsInteger(8),
    'd0_1' / construct.BitsInteger(8),
    'd0_2' / construct.BitsInteger(8),
    'd0_3' / construct.BitsInteger(8),
    'd1_0' / construct.BitsInteger(8),
    'd1_1' / construct.BitsInteger(8),
    'd1_2' / construct.BitsInteger(8),
    'd1_3' / construct.BitsInteger(8))
def get_snap_adc(c, fpgas = [], wait_period = 3):
    """
    Read raw samples from the ADC snap block.
    2 pols, each one 4 parallel samples f8.7. So 64-bits total.
    """
    raw = snap.snapshots_get(fpgas = fpgas, dev_names = snap_adc, wait_period = wait_period)
    repeater = construct.GreedyRepeater(snap_fengine_adc)
    rv = []
    for index, d in enumerate(raw['data']):
        upd = repeater.parse(d)
        data = [[], []]
        for ctr in range(0, len(upd)):
            for pol in range(0,2):
                for sample in range(0,4):
                    uf = upd[ctr]['d%i_%i' % (pol,sample)]
                    f87 = bin2fp(uf)
                    data[pol].append(f87)
        v = {'fpga_index': index, 'data': data}
        rv.append(v)
    return rv
def get_snap_adc_DUMB(c, fpgas = [], wait_period = 3):
    """
    Read raw samples from the ADC snap block.
    2 pols, each one 4 parallel samples f8.7. So 64-bits total.
    """
    raw = snap.snapshots_get(fpgas = fpgas, dev_names = snap_adc, wait_period = wait_period)
    repeater = construct.GreedyRepeater(snap_fengine_adc)
    rv = []
    for index, d in enumerate(raw['data']):
        data = [[],[]]
        od = numpy.fromstring(d, dtype = numpy.int8)
        for ctr in range(0, len(od), 8):
            for ctr2 in range(0,4):
                data[0].append(od[ctr + ctr2])
            for ctr2 in range(4,8):
                data[1].append(od[ctr + ctr2])
        data = [numpy.array(data[0], dtype=numpy.int8), numpy.array(data[1], dtype=numpy.int8)]
        v = {'fpga_index': index, 'data': data}
        rv.append(v)
    return rv
def get_adc_snapshot(c, ant_names, trig_level = -1, sync_to_pps = True):
    if (trig_level >= 0) or (sync_to_pps == False):
        raise RuntimeError('Not currently supported. Soon, Captain, soon...')

    # horrid horrid translation step because of that KAK way data is organised in this package
    fpgas = []
    ffpga_numbers = []
    ant_details = {}
    index = 0
    for ant_str in ant_names:
        (ffpga_n, xfpga_n, fxaui_n, xxaui_n, feng_input) = c.get_ant_str_location(ant_str)
        f = c.ffpgas[ffpga_n]
        if fpgas.count(f) == 0:
            fpgas.append(f)
            ffpga_numbers.append(ffpga_n)
            ant_details[ffpga_n] = {}
        ant_details[ffpga_n][feng_input] = ant_str
    # get the data
    data = get_snap_adc_DUMB(c, fpgas = fpgas)
    # mangle it to return it
    rv = {}
    for n, d in enumerate(data):
        for p, poldata in enumerate(d['data']):
            t = {}
            t['timestamp'] = 0
            t['data'] = poldata
            t['length'] = len(poldata)
            t['offset'] = 0
            astr = None
            try:
                fnum = ffpga_numbers[n]
                astr = ant_details[fnum][p]
            except KeyError:
                pass
            if astr != None:
                rv[astr] = t
    return rv

snap_fengine_debug_select = {}
snap_fengine_debug_select['coarse_72'] =    0
snap_fengine_debug_select['fine_128'] =     1
snap_fengine_debug_select['quant_16'] =     2
snap_fengine_debug_select['ct_64'] =        3
snap_fengine_debug_select['xaui_128'] =     4
snap_fengine_debug_select['gbetx0_128'] =   5
snap_fengine_debug_select['buffer_72'] =    6
snap_fengine_debug_select['pfb_72'] =       7

snap_fengine_debug_coarse_fft = snap_debug / construct.BitStruct(construct.Padding(128 - (4*18)),
    'd0_r' / construct.BitsInteger(18),
    'd0_i' / construct.BitsInteger(18),
    'd1_r' / construct.BitsInteger(18),
    'd1_i' / construct.BitsInteger(18))
def get_snap_coarse_fft(c, fpgas = [], pol = 0, setup_snap = True):
    """
    Read and return data from the coarse FFT.
    Returns a list of the data from only that polarisation.
    """
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    if setup_snap:
        corr_functions.write_masked_register(fpgas, register_fengine_control,           debug_snap_select = snap_fengine_debug_select['coarse_72'])
        corr_functions.write_masked_register(fpgas, register_fengine_coarse_control,    debug_pol_select = pol, debug_specify_chan = 0)
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3)
    rd = []
    for ctr in range(0, len(snap_data['data'])):
        d = snap_data['data'][ctr]
        repeater = construct.GreedyRepeater(snap_fengine_debug_coarse_fft)
        up = repeater.parse(d)
        coarsed = []
        for a in up:
            for b in range(0,2):
                num = bin2fp(a['d%i_r'%b], 18, 17) + (1j * bin2fp(a['d%i_i'%b], 18, 17))
                coarsed.append(num)
        rd.append(coarsed)
    return rd

def get_snap_coarse_channel(c, fpgas = [], pol = 0, channel = -1, setup_snap = True):
    """
    Get data from a specific coarse channel - straight out of the FFT into the snap block, NOT via the buffer block.
    Returns a list of the data from only that polarisation.
    """
    if channel == -1:
        raise RuntimeError('Cannot get data from unspecified channel.')
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    if setup_snap:
        corr_functions.write_masked_register(fpgas, register_fengine_control,           debug_snap_select = snap_fengine_debug_select['coarse_72'])
        corr_functions.write_masked_register(fpgas, register_fengine_coarse_control,    debug_pol_select = pol, debug_specify_chan = 1, debug_chan = channel >> 1)
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3)
    rd = []
    for ctr in range(0, len(snap_data['data'])):
        d = snap_data['data'][ctr]
        repeater = construct.GreedyRepeater(snap_fengine_debug_coarse_fft)
        up = repeater.parse(d)
        coarsed = []
        for a in up:
            if channel & 1:
                num = bin2fp(a['d1_r'], 18, 17) + (1j * bin2fp(a['d1_i'], 18, 17))
            else:
                num = bin2fp(a['d0_r'], 18, 17) + (1j * bin2fp(a['d0_i'], 18, 17))
            coarsed.append(num)
        rd.append(coarsed)
    return rd

def get_snap_buffer_pfb(c, fpgas = [], pol = 0, setup_snap = True, pfb = False):
    '''This DOESN'T EXIST in regular F-engines. Only in specific debug versions.
    '''
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    if setup_snap:
        if pfb:
            corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['pfb_72'])
        else:
            corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['buffer_72'])
        corr_functions.write_masked_register(fpgas, register_fengine_coarse_control, debug_pol_select = pol)
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3)
    rd = []
    for ctr in range(0, len(snap_data['data'])):
        d = snap_data['data'][ctr]
        repeater = construct.GreedyRepeater(snap_fengine_debug_coarse_fft)
        up = repeater.parse(d)
        coarsed = []
        for a in up:
            num = bin2fp(a['d%i_r'%pol], 18, 17) + (1j * bin2fp(a['d%i_i'%pol], 18, 17))
            coarsed.append(num)
        rd.append(coarsed)
    return rd

#snap_fengine_debug_fine_fft = construct.BitStruct(snap_debug,
#    construct.Padding(128 - 72),
#    construct.BitField("p0_r", 18),
#    construct.BitField("p0_i", 18),
#    construct.BitField("p1_r", 18),
#    construct.BitField("p1_i", 18))
#snap_fengine_debug_fine_fft_tvg = construct.BitStruct(snap_debug,
#    construct.Padding(128-32),
#    construct.BitField("ctr", 32))
#def get_snap_fine_tvg(c, fpgas, offset = -1):
#    if len(fpgas) == 0:
#        fpgas = c.ffpgas
#    corr_functions.write_masked_register(fpgas, register_fengine_fine_control, fine_debug_select = 2)
#    corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = 1, tvg_en = True, fine_tvg = True)
#    import time
#    time.sleep(1)
#    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3, offset = offset)
#    rd = []
#    for ctr in range(0, len(snap_data['data'])):
#        d = snap_data['data'][ctr]
#        repeater = construct.GreedyRepeater(snap_fengine_debug_fine_fft_tvg)
#        up = repeater.parse(d)
#        fdata = []
#        for a in up:
#            p0c = a['ctr']
#            fdata.append(p0c)
#        rd.append(fdata)
#    return rd
fine_fft_bitwidth = 31;
snap_fengine_debug_fine_fft = snap_debug / construct.BitStruct(construct.Padding(128 - (4*fine_fft_bitwidth)),
    'p0_r' / construct.BitsInteger(fine_fft_bitwidth),
    'p0_i' / construct.BitsInteger(fine_fft_bitwidth),
    'p1_r' / construct.BitsInteger(fine_fft_bitwidth),
    'p1_i' / construct.BitsInteger(fine_fft_bitwidth))
def get_snap_fine_fft(c, fpgas = [], offset = -1, setup_snap = True):
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    if setup_snap:
        corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['fine_128'])
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3, offset = offset)
    rd = []
    for ctr in range(0, len(snap_data['data'])):
        d = snap_data['data'][ctr]
        repeater = construct.GreedyRepeater(snap_fengine_debug_fine_fft)
        up = repeater.parse(d)
        fdata_p0 = []
        fdata_p1 = []
        for a in up:
            p0c = bin2fp(a['p0_r'], fine_fft_bitwidth, 17) + (1j * bin2fp(a['p0_i'], fine_fft_bitwidth, 17))
            p1c = bin2fp(a['p1_r'], fine_fft_bitwidth, 17) + (1j * bin2fp(a['p1_i'], fine_fft_bitwidth, 17))
            fdata_p0.append(p0c)
            fdata_p1.append(p1c)
        rd.append([fdata_p0, fdata_p1])
    return rd

snap_fengine_debug_quant = snap_debug / construct.BitStruct(construct.Padding(128 - 16),
    'p0_r' / construct.BitsInteger(4),
    'p0_i' / construct.BitsInteger(4),
    'p1_r' / construct.BitsInteger(4),
    'p1_i' / construct.BitsInteger(4))
def get_snap_quant_wbc_compat(c, fpgas = [], offset = -1):
    return get_snap_quant(c = c, fpgas = fpgas, offset = offset, wbc_compat = True)
def get_snap_quant(c, fpgas = None, offset = -1, wbc_compat = False, debug_data = None, setup_snap = True):
    """
    Read and return data from the quantiser snapshot. Both pols are returned.
    """
    if fpgas == None:
        fpgas = c.ffpgas
    if setup_snap:
        c.syslogger.debug('get_snap_quant: setting debug snapblock to quantiser output.')
        corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['quant_16'])
    data = []
    for fpga in fpgas:
        tempdata = _fpga_snap_quant(fpga = fpga, offset = offset, wbc_compat = wbc_compat, debug_data = debug_data)
        data.append(tempdata)
    return data

def _fpga_snap_quant(fpga = None, offset = -1, wbc_compat = False, debug_data = None):
    ''''
    Get quantiser snap data from only one f-engine FPGA.
    NB: Assumes the quantiser has already been selected in the control register.
    Returns a snapshot of quantised data in one of two formats, depending on the wbc_compat argument.
    Either way, it's data for both pols.
    debug_data is data from the snap.snapshots_get function
    '''
    def _log(msg):
        fpga._logger.debug('_fpga_snap_quant: %s' % msg)
    if fpga == None:
        raise RuntimeError('Please provide the FPGA from which to read the quantised data.')
    if debug_data == None:
        _log('reading snap data at offset %i.' % offset)
        snap_data = snap.snapshots_get(fpgas = [fpga], dev_names = snap_debug, wait_period = 3, offset = offset)['data'][0]
    else:
        _log('using debug data, not fresh snap data.')
        snap_data = debug_data['data'][0]
    _log('unpacking data.')
    data = [[], []]
    if not wbc_compat:
        repeater = construct.GreedyRepeater(snap_fengine_debug_quant)
        unpacked = repeater.parse(snap_data)
        for ctr in unpacked:
            p0c = bin2fp(ctr['p0_r'], 4, 3) + (1j * bin2fp(ctr['p0_i'], 4, 3))
            p1c = bin2fp(ctr['p1_r'], 4, 3) + (1j * bin2fp(ctr['p1_i'], 4, 3))
            data[0].append(p0c)
            data[1].append(p1c)
    else:
        # remember that the data is 16-bit padded up to 128-bit because of the one debug snap block, so only 2 of every 16 bytes are valid data
        unpacked = numpy.fromstring(snap_data, dtype = numpy.uint8)
        for ctr in range(14, len(unpacked), 16):
            pol0_r_bits = (unpacked[ctr]   & ((2**8) - (2**4))) >> 4
            pol0_i_bits = (unpacked[ctr]   & ((2**4) - (2**0)))
            pol1_r_bits = (unpacked[ctr+1] & ((2**8) - (2**4))) >> 4
            pol1_i_bits = (unpacked[ctr+1] & ((2**4) - (2**0)))
            data[0].append(float(((numpy.int8(pol0_r_bits << 4) >> 4))) + (1j * float(((numpy.int8(pol0_i_bits << 4) >> 4)))))
            data[1].append(float(((numpy.int8(pol1_r_bits << 4) >> 4))) + (1j * float(((numpy.int8(pol1_i_bits << 4) >> 4)))))
    _log('returning %i complex values for each pol.' % len(data[0]))
    return data

def get_quant_spectrum(c, fpgas = None):
    raise RuntimeError('NOT YET COMPLETE. SORRY.')
    num_chans = c.config['n_chans']
    rv = []
    spectrum = 3 
    return spectrum

snap_fengine_debug_ct = snap_debug / construct.BitStruct(construct.Padding(128 - 64),
    'p00_r' / construct.BitsInteger(4), 'p00_i' / construct.BitsInteger(4), 'p10_r' / construct.BitsInteger(4), 'p10_i' / construct.BitsInteger(4),
    'p01_r' / construct.BitsInteger(4), 'p01_i' / construct.BitsInteger(4), 'p11_r' / construct.BitsInteger(4), 'p11_i' / construct.BitsInteger(4),
    'p02_r' / construct.BitsInteger(4), 'p02_i' / construct.BitsInteger(4), 'p12_r' / construct.BitsInteger(4), 'p12_i' / construct.BitsInteger(4),
    'p03_r' / construct.BitsInteger(4), 'p03_i' / construct.BitsInteger(4), 'p13_r' / construct.BitsInteger(4), 'p13_i' / construct.BitsInteger(4))
def get_snap_ct(c, fpgas = [], offset = -1, setup_snap = True):
    """
    Read and return data from the corner turner. Both pols are returned.
    """
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    if setup_snap:
        corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['ct_64'])
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3, offset = offset)
    rd = []
    for ctr in range(0, len(snap_data['data'])):
        d = snap_data['data'][ctr]
        repeater = construct.GreedyRepeater(snap_fengine_debug_ct)
        up = repeater.parse(d)
        fdata_p0 = []
        fdata_p1 = []
        for a in up:
            p0 = []
            p1 = []
            p0.append(bin2fp(a['p00_r'], 4, 3) + (1j * bin2fp(a['p00_i'], 4, 3)))
            p0.append(bin2fp(a['p01_r'], 4, 3) + (1j * bin2fp(a['p01_i'], 4, 3)))
            p0.append(bin2fp(a['p02_r'], 4, 3) + (1j * bin2fp(a['p02_i'], 4, 3)))
            p0.append(bin2fp(a['p03_r'], 4, 3) + (1j * bin2fp(a['p03_i'], 4, 3)))
            p1.append(bin2fp(a['p10_r'], 4, 3) + (1j * bin2fp(a['p10_i'], 4, 3)))
            p1.append(bin2fp(a['p11_r'], 4, 3) + (1j * bin2fp(a['p11_i'], 4, 3)))
            p1.append(bin2fp(a['p12_r'], 4, 3) + (1j * bin2fp(a['p12_i'], 4, 3)))
            p1.append(bin2fp(a['p13_r'], 4, 3) + (1j * bin2fp(a['p13_i'], 4, 3)))
            fdata_p0.extend(p0)
            fdata_p1.extend(p1)
        rd.append([fdata_p0, fdata_p1])
    return rd

# the xaui snap block on the f-engine - this is just after packetisation
snap_fengine_xaui = 'snap_debug' / construct.BitStruct(construct.Padding(128 - 1 - 3 - 1 - 1 - 3 - 64),
    'link_down' / construct.Flag,
    construct.Padding(3),
    'mrst' / construct.Flag,
    construct.Padding(1),
    'eof' / construct.Flag,
    'sync' / construct.Flag,
    'hdr_valid' / construct.Flag,
    'data' / construct.BitsInteger(64))
def get_snap_xaui(c, fpgas = [], offset = -1, man_trigger = False, man_valid = False, wait_period = 3):
    """
    Read the XAUI data out of the general debug snap block.
    """
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['xaui_128'])
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = wait_period, offset = offset, man_trig = man_trigger, man_valid = man_valid, circular_capture = False)
    return snap_data

snap_fengine_gbe_tx = 'snap_debug' / construct.BitStruct(construct.Padding(128 - 64 - 32 - 6),  
    'eof' / construct.Flag, 
    'link_up' / construct.Flag, 
    'led_tx' / construct.Flag, 
    'tx_full' / construct.Flag, 
    'tx_over' / construct.Flag, 
    'valid' / construct.Flag,
    'ip_addr' / construct.BitsInteger(32),
    'data' / construct.BitsInteger(64))
def get_snap_feng_10gbe(c, fpgas = [], offset = -1,  man_trigger = False, man_valid = False):
    if len(fpgas) == 0:
        fpgas = c.ffpgas
    corr_functions.write_masked_register(fpgas, register_fengine_control, debug_snap_select = snap_fengine_debug_select['gbetx0_128'])
    snap_data = snap.snapshots_get(fpgas = fpgas, dev_names = snap_debug, wait_period = 3, offset = offset, man_trig = man_trigger, man_valid = man_valid, circular_capture = False)
    rd = []
    for ctr in range(0, len(snap_data['data'])):
        d = snap_data['data'][ctr]
        repeater = construct.GreedyRepeater(snap_fengine_gbe_tx)
        up = repeater.parse(d)
        for a in up:
            a['link_down'] = not a['link_up']
            a['hdr_valid'] = False
            a['mrst'] = False
            a['sync'] = False
        rd.append(up)
    return rd

def DONE_get_fine_fft_snap(correlator):
    # interpret the ant_string
    (ffpga_n, xfpga_n, fxaui_n, xxaui_n, feng_input) = correlator.get_ant_str_location(ant_str)
    # select the data from the fine fft
    fpga = correlator.ffpgas[ffpga_n]
    corr_functions.write_masked_register([fpga], register_fengine_fine_control, snap_data_select = 0, quant_snap_select = 0)
    data = fpga.snapshot_get(dev_name = fine_snap_name, man_trig = False, man_valid = False, wait_period = 3, offset = -1, circular_capture = False, get_extra_val = False)
    unpacked = list(struct.unpack('>%iI' % (len(data['data']) / 4), data['data']))
    # re-arrange the data sensibly - for FFT data it's complex 16.15 fixed point signed data
    # make the actual complex numbers
    d  = []
    for ctr in range(0, len(unpacked)):
        num = unpacked[ctr]
        numR = numpy.int16(num >> 16)
        numI = numpy.int16(num & 0x0000ffff)
        d.append(numR + (1j * numI))
    return d

def DONE_get_ct_snap(correlator, offset = -1):
    corr_functions.write_masked_register(correlator.ffpgas, register_fengine_fine_control, quant_snap_select = 2)
    raw = snap.snapshots_get(correlator.ffpgas, dev_names = fine_snap_name, man_trig = False, man_valid = False, wait_period = 3, offset = offset, circular_capture = False)
    chan_values = []
    for index, d in enumerate(raw['data']):
        up = list(struct.unpack('>%iI' % (len(d) / 4), d))
        values = [[], []]
        for value in up:
            # two freq channel values for the same freq-channel, both pols
            # will have to use the offset to get multiple freq channels
            p00 = value >> 24
            p10 = (value >> 16) & 0xff
            p01 = (value >> 8) & 0xff
            p11 = value & 0xff
            def extract8bit(v8):
                r = (v8 & ((2**8) - (2**4))) >> 4
                i = (v8 & ((2**4) - (2**0)))
                return r + (1j * i)
            values[0].append(value >> 24)
            values[0].append((value >> 8) & 0xff)
            values[1].append((value >> 16) & 0xff)
            values[1].append(value & 0xff)
        chan_values.append({'fpga_index': index, 'data': values})
    return chan_values
# end
