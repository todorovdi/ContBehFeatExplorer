#!/usr/bin/python3
from mpi4py import MPI
comm = MPI.COMM_WORLD
my_rank = comm.Get_rank()
p = comm.Get_size()

import sys

print(my_rank)

if my_rank == 0:
    print(my_rank,sys.argv)
# get parameters in process zero only, load features, broadcast it to everyone
# send particular small params to individ processes then run main part of
# run_PCA
# evey process separately saves the result

MPI.Finalize
