#!/bin/bash
# This is horrible. All directories are hard coded
# rsync takes flags, directory to back up, destination to back up to
# this takes data from termTest and backs up to both RAID on DR2 and
# peloton using jmlev's account
sudo rsync --progress -avzu /drBiggerBoy/termTest_3_29_23 /media/dradmin/RAID\ 1/drBackups/run1p3_4_19_23
rsync -avz --progress -e ssh /drBiggerBoy/termTest_3_29_23 jmlev@peloton.cse.ucdavis.edu:/group/tysongrp/run1p3_4_19_23

