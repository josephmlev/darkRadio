#!/bin/bash
# This is horrible. All directories are hard coded
# rsync takes flags, directory to back up, destination to back up to
# this takes data from termTest and backs up to both RAID on DR2 and
# peloton using jmlev's account

# injection data to RAID
sudo rsync --progress -avzu /drBiggerBoy/run1p3/injection_data  /media/dradmin/RAID\ 1/drBackups/run1p3_4_19_23/

# injection data to Peloton
rsync -avz --progress -e ssh /drBiggerBoy/run1p3/injection_data jmlev@peloton.cse.ucdavis.edu:/group/tysongrp/run1p3_4_19_23/

# main data to RAID
sudo rsync --progress -avzu /drBiggerBoy/run1p3/main_data  /media/dradmin/RAID\ 1/drBackups/run1p3_4_19_23/

# main data to Peloton
rsync -avz --progress -e ssh /drBiggerBoy/run1p3/main_data jmlev@peloton.cse.ucdavis.edu:/group/tysongrp/run1p3_4_19_23/

