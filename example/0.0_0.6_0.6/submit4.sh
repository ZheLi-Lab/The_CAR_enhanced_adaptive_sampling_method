#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
##Production
PRMTOP=`ls ../*.prmtop|head -n 1`

pmemd.cuda_SPFP -O -i prod4.in -p $PRMTOP -c prev_cen.rst -o prod4.out -r prod4.rst -inf prod4.mdinfo -x prod4.netcdf -AllowSmallBox
