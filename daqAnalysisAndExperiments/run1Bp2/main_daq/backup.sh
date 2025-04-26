#!/bin/bash
# This is horrible. All directories are hard coded
# rsync takes flags, directory to back up, destination to back up to
# this takes data from termTest and backs up to both RAID on DR2 and
# peloton using jmlev's account

# main data to RAID
sudo rsync --progress -avzu /drBiggerBoy/run1Bp1  /media/dradmin/RAID\ 1/drBackups/run1Bp1_9_14_23/

# main data to dr3
rsync -avz --progress -e ssh /drBiggerBoy/run1Bp1 dark-radio@darkradio3.physics.ucdavis.edu:/RunData/run1Bp1_9_14_23/

