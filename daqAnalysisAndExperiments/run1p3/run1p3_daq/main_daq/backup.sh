#!/bin/bash
# This is horrible. All directories are hard coded
# rsync takes flags, directory to back up, destination to back up to
# this takes data from termTest and backs up to both RAID on DR2 and
# peloton using jmlev's account

# injection data to RAID
sudo rsync --progress -avzu /drBiggerBoy/run1p3/injection_data  /media/dradmin/RAID\ 1/drBackups/run1p3_4_19_23/

# injection data to dr3
rsync -avz --progress -e ssh /drBiggerBoy/run1p3/injection_data dark-radio@darkradio3.physics.ucdavis.edu:/RunData/run1p3_4_19_23/

# main data to RAID
sudo rsync --progress -avzu /drBiggerBoy/run1p3/main_data  /media/dradmin/RAID\ 1/drBackups/run1p3_4_19_23/

# main data to dr3
rsync -avz --progress -e ssh /drBiggerBoy/run1p3/main_data dark-radio@darkradio3.physics.ucdavis.edu:/RunData/run1p3_4_19_23/

