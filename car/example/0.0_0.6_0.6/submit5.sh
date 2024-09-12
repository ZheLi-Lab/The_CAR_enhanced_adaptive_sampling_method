#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
##Production
PRMTOP=`ls ../*.prmtop|head -n 1`

pmemd.cuda_SPFP -O -i prod5.in -p $PRMTOP -c prev_cen.rst -o prod5.out -r prod5.rst -inf prod5.mdinfo -x prod5.netcdf -AllowSmallBox
