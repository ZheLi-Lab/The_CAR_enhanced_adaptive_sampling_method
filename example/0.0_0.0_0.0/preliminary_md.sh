#!/bin/bash
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
#export CUDA_VISIBLE_DEVICES=0

PRMTOP=`ls ../*.prmtop|head -n 1`
PRMCRD=`ls ../*.prmcrd|head -n 1`
####min.sh
pmemd.cuda_SPFP -O -i min.in  -p $PRMTOP -c $PRMCRD  -o min.out  -r min.rst  -inf min.mdinfo  -x min.netcdf  -ref $PRMCRD -AllowSmallBox

####heat.sh
pmemd -O -i heat-cpu.in -p $PRMTOP -c min.rst  -o heat-cpu.out -r heat-cpu.rst -inf heat-cpu.mdinfo -x heat-cpu.netcdf -ref min.rst -AllowSmallBox
pmemd.cuda_SPFP -O -i heat.in -p $PRMTOP -c heat-cpu.rst  -o heat.out -r heat.rst -inf heat.mdinfo -x heat.netcdf -ref heat-cpu.rst -AllowSmallBox

####equi.sh
pmemd.cuda_SPFP -O -i equi-pre.in -p $PRMTOP -c heat.rst -o equi-1.out -r equi-1.rst -inf equi-1.mdinfo -x equi-1.netcdf -ref heat.rst -AllowSmallBox
pmemd.cuda_SPFP -O -i equi-pre.in -p $PRMTOP -c equi-1.rst -o equi-2.out -r equi-2.rst -inf equi-2.mdinfo -x equi-2.netcdf -ref equi-1.rst -AllowSmallBox
pmemd.cuda_SPFP -O -i equi-pre.in -p $PRMTOP -c equi-2.rst -o equi-3.out -r equi-3.rst -inf equi-3.mdinfo -x equi-3.netcdf -ref equi-2.rst -AllowSmallBox
pmemd.cuda_SPFP -O -i equi-pre.in -p $PRMTOP -c equi-3.rst -o equi-4.out -r equi-4.rst -inf equi-4.mdinfo -x equi-4.netcdf -ref equi-3.rst -AllowSmallBox
pmemd.cuda_SPFP -O -i equi.in -p $PRMTOP -c equi-4.rst -o equi-5.out -r equi-5.rst -inf equi-5.mdinfo -x equi-5.netcdf -ref equi-4.rst -AllowSmallBox
