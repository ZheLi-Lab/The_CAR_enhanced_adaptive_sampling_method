#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
##Production
PRMTOP=`ls ../*.prmtop|head -n 1`

pmemd.cuda_SPFP -O -i prod0.in -p $PRMTOP -c prev_cen.rst -o prod0.out -r prod0.rst -inf prod0.mdinfo -x prod0.netcdf -AllowSmallBox
