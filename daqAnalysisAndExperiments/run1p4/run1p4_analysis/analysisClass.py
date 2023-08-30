import numpy as np

class AnalysisPipeline:
    """
    A class to perform a dark photon search on an averaged spectrum

    Attributes
    ----------
    spec            : arr
        Power spectrum to search for dark photon signal
    freqs           : arr
        Array of frequenies that corosponds to spec
    analysisSpec    : arr
        Spectrum that is being worked on

    Methods
    -------
    high_pass_filter(fcNumBins=int,
        order=int):
        Pre-filters the spectrum to remove DC offset and large scale undulations

    optimal_filter(
        ):
        Performs a convolution of the input against a template array
        Calls gen_template

    gen_template(
        ):
        Generates a template function array
    
    rolling_STD(window=int,
        filt=bool,
        fcNumBins=int,
        order=int,
        numProc=int (opt.),
        mode=str
        ):
        Calculates rolling 1 STD threshold using an optional function

    find_candidates(CL=float,
        ):
        Calculates a threshold such that there is a (1-CL)% chance
        that a bin is above that threshold.
    """
    def __init__(self,
        spec, 
        freqs
        ):

        self.spec = spec
        self.freqs = freqs
        self.analysisSpec = np.zeros(spec)

    def high_pass_filter(self):
        '''
        Performs basic buttersworth filtering of a spectrum.
            Parameters:
                fcNumBins (int) : Corner "frequency" in bins
                order (int)     : Order of filter
        '''
        pass

    def optimal_filter(self):
        """
        Implement optimal filtering/convolution here
        """
        pass

    def rolling_STD(self):
        '''
        Computes a rolling STD on the input spectrum. 
            Parameters:
                window (int)    : length of window to compute rolling function on
                filt (bool)     : should the output be filtered. 1 = yes
                fcNumBins (int) : if filt, corner "frequency" (in bins)
                order (int)     : if filt, order of filter
                mode (str)      : how to handle edges
                                'nan pad'
                                    preserve length of input array. Pad ends with nands
                numProc (int)   : number of processers to use
                
        '''
        pass

    def calculate_threshold(self):
        """
        Implement threshold calculation here
        """
        pass

    def run_analysis(self):
        self.high_pass_filter()
        self.optimal_filter()
        self.calculate_rolling_mad()
        self.calculate_threshold()


######## Call this from notebook ########
# Instantiate the analysis pipeline
#analysis = AnalysisPipeline(spectrum, window_size, step_size, num_processes)

# Run the full analysis
#analysis.run_analysis()