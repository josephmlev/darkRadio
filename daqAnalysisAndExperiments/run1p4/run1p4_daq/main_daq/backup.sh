#!/bin/bash
# This is horrible. All directories are hard coded
# rsync takes flags, directory to back up, destination to back up to
# this takes data from termTest and backs up to both RAID on DR2 and
# peloton using jmlev's account

# main data to RAID
sudo rsync --progress -avzu /drBiggerBoy/run1p4  /media/dradmin/RAID\ 1/drBackups/run1p4_5_10_23/

# main data to dr3
rsync -avz --progress -e ssh /drBiggerBoy/run1p4 dark-radio@darkradio3.physics.ucdavis.edu:/RunData/run1p4_5_10_23/

