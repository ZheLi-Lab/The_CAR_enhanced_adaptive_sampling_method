#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
##Production
PRMTOP=`ls ../*.prmtop|head -n 1`

pmemd.cuda_SPFP -O -i prod1.in -p $PRMTOP -c prev_cen.rst -o prod1.out -r prod1.rst -inf prod1.mdinfo -x prod1.netcdf -AllowSmallBox
