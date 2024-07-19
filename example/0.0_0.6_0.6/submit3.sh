#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
##Production
PRMTOP=`ls ../*.prmtop|head -n 1`

pmemd.cuda_SPFP -O -i prod3.in -p $PRMTOP -c prev_cen.rst -o prod3.out -r prod3.rst -inf prod3.mdinfo -x prod3.netcdf -AllowSmallBox
