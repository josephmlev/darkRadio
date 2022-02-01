"""
Setup and unique functionality for the wide-band correlator modes. A wideband correlator's FPGAs process all digitised data, which is a multiple of the FPGA clock rates.
"""
"""
Revisions:
2011-07-07  PVP  Initial revision.
"""
import construct, corr_functions

# f-engine control register
#'control'
register_fengine_control = 'control' / construct.BitStruct(construct.Padding(32 - 20 - 1),             # 21 - 31
    'tvgsel_noise' / construct.Flag,
    'tvgsel_fdfs' / construct.Flag,
    'tvgsel_pkt' / construct.Flag,
    'tvgsel_ct' / construct.Flag,
    'tvg_en' / construct.Flag,
    construct.Padding(16 - 13 - 1),
    'adc_protect_disable' / construct.Flag,
    'flasher_en' / construct.Flag,
    construct.Padding(12 - 9 - 1),
    'gbe_enable' / construct.Flag,
    'gbe_rst' / construct.Flag,
    construct.Padding(8 - 3 - 1),
    'clr_status' / construct.Flag,
    'arm' / construct.Flag,
    'soft_sync' / construct.Flag,
    'mrst' / construct.Flag)

    #construct.Flag('tvgsel_noise'),             # 20
    #construct.Flag('tvgsel_fdfs'),              # 19
    #construct.Flag('tvgsel_pkt'),               # 18
    #construct.Flag('tvgsel_ct'),                # 17
    #construct.Flag('tvg_en'),                   # 16
    #construct.Padding(16 - 13 - 1),             # 14 - 15
    #construct.Flag('adc_protect_disable'),      # 13
    #construct.Flag('flasher_en'),               # 12
    #construct.Padding(12 - 9 - 1),              # 10 - 11
    #construct.Flag('gbe_enable'),               # 9
    #construct.Flag('gbe_rst'),                  # 8
    #construct.Padding(8 - 3 - 1),               # 4 - 7
    #construct.Flag('clr_status'),               # 3
    #construct.Flag('arm'),                      # 2
    #construct.Flag('soft_sync'),                # 1
    #construct.Flag('mrst'))                     # 0

# f-engine status
#'fstatus0',
register_fengine_fstatus = 'fstatus0' / construct.BitStruct(construct.Padding(32 - 29 - 1),     # 30 - 31
    'sync_val' / construct.BitsInteger(2),  # 28 - 29
    construct.Padding(28 - 17 - 1),     # 18 - 27
    'xaui_lnkdn' / construct.Flag,       # 17
    'xaui_over' / construct.Flag,        # 16
    construct.Padding(16 - 6 - 1),      # 7 - 15
    'dram_err' / construct.Flag,         # 6
    'clk_err' / construct.Flag,          # 5
    'adc_disabled' / construct.Flag,     # 4
    'ct_error' / construct.Flag,         # 3
    'adc_overrange' / construct.Flag,    # 2
    'fft_overrange' / construct.Flag,    # 1
    'quant_overrange' / construct.Flag) # 0

# x-engine control register
#'ctrl'
register_xengine_control = 'ctrl' / construct.BitStruct(construct.Padding(32 - 16 - 1),     # 17 - 31
    'gbe_out_enable' / construct.Flag,   # 16
    'gbe_rst' / construct.Flag,          # 15
    construct.Padding(15 - 12 - 1),     # 13 - 14
    'flasher_en' / construct.Flag,       # 12
    'gbe_out_rst' / construct.Flag,      # 11
    'loopback_mux_rst' / construct.Flag, # 10
    'gbe_enable' / construct.Flag,       # 9
    'cnt_rst' / construct.Flag,          # 8
    'clr_status' / construct.Flag,       # 7
    construct.Padding(7 - 0 - 1),       # 1 - 6
    'vacc_rst' / construct.Flag)         # 0

# x-engine status
#'xstatus0'
register_xengine_status = 'xstatus0' / construct.BitStruct(construct.Padding(32 - 18 - 1),     # 19 - 31
    'gbe_lnkdn' / construct.Flag,        # 18
    'xeng_err' / construct.Flag,         # 17
    construct.Padding(17 - 5 - 1),      # 6 - 16
    'vacc_err' / construct.Flag,         # 5
    'rx_bad_pkt' / construct.Flag,       # 4
    'rx_bad_frame' / construct.Flag,     # 3
    'tx_over' / construct.Flag,          # 2
    'pkt_reord_err' / construct.Flag,    # 1
    'pack_err' / construct.Flag)        # 0

# x-engine tvg control
#'tvg_sel'
register_xengine_tvg_sel = 'tvg_sel' / construct.BitStruct(construct.Padding(32 - 1 - 2 - 2 - 6),  # 11 - 31
    'vacc_tvg_sel' / construct.BitsInteger(6),  # 5 - 10
    'xeng_tvg_sel' / construct.BitsInteger(2),  # 3 - 4
    'descr_tvg_sel' / construct.BitsInteger(2), # 1 - 2
    'xaui_tvg_sel' / construct.Flag)         # 0

# "snap_rx0"
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

# "snap_gbe_rx0"
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

# "snap_gbe_tx0"
snap_xengine_gbe_tx = 'snap_gbe_tx0' / construct.BitStruct(construct.Padding(128 - 64 - 32 - 6), 
        'eof' / construct.Flag,
        'link_up' / construct.Flag,
        'led_tx' / construct.Flag,
        'tx_full' / construct.Flag,
        'tx_over' / construct.Flag,
        'valid' / construct.Flag,
        'ip_addr' / construct.BitsInteger(32),
        'data' / construct.BitsInteger(64))

# the snap block immediately after the x-engine
# "snap_vacc0"
snap_xengine_vacc = 'snap_vacc0' / construct.BitStruct('data' / construct.BitsInteger(32))

# the xaui snap block on the f-engine - this is just after packetisation
snap_fengine_xaui = 'snap_xaui0' / construct.BitStruct(construct.Padding(128 - 1 - 3 - 1 - 1 - 3 - 64),
    'link_down' / construct.Flag,
    construct.Padding(3),
    'mrst' / construct.Flag,
    construct.Padding(1),
    'eof' / construct.Flag,
    'sync' / construct.Flag,
    'hdr_valid' / construct.Flag,
    'data' / construct.BitsInteger(64))

snap_fengine_gbe_tx = 'snap_gbe_tx0' / construct.BitStruct(construct.Padding(128 - 64 - 32 - 6), 
    'eof' / construct.Flag,
    'link_up' / construct.Flag,
    'led_tx' / construct.Flag,
    'tx_full' / construct.Flag,
    'tx_over' / construct.Flag,
    'valid' / construct.Flag,
    'ip_addr' / construct.BitsInteger(32),
    'data' / construct.BitsInteger(64))


def feng_status_get(c, ant_str):
    """Reads and decodes the status register for a given antenna. Adds some other bits 'n pieces relating to Fengine status."""
    #'sync_val': 28:30, #This is the number of clocks of sync pulse offset for the demux-by-four ADC 1PPS.
    ffpga_n, xfpga_n, fxaui_n, xxaui_n, feng_input = c.get_ant_str_location(ant_str)
    rv = corr_functions.read_masked_register([c.ffpgas[ffpga_n]], register_fengine_fstatus, names = ['fstatus%i' % feng_input])[0]
    if rv['xaui_lnkdn'] or rv['xaui_over'] or rv['clk_err'] or rv['ct_error'] or rv['fft_overrange']:
        rv['lru_state']='fail'
    elif rv['adc_overrange'] or rv['adc_disabled']:
        rv['lru_state']='warning'
    else:
        rv['lru_state']='ok'
    return rv

# end
