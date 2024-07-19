#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0
##Production
PRMTOP=`ls ../*.prmtop|head -n 1`

pmemd.cuda_SPFP -O -i prod2.in -p $PRMTOP -c prev_cen.rst -o prod2.out -r prod2.rst -inf prod2.mdinfo -x prod2.netcdf -AllowSmallBox
