#!/usr/bin/python3
import joblib
from joblib import Parallel, parallel_backend
from joblib import register_parallel_backend
from joblib import delayed
from ipyparallel import Client
from ipyparallel.joblib import IPythonParallelBackend
import os,sys
import argparse
import logging

from joblib import cpu_count
import multiprocessing as mpr

FILE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(FILE_DIR)

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--profile", default="ipy_profile",
                 help="Name of IPython profile to use")
args = parser.parse_args()
profile = args.profile


#prepare the engines
client = Client(profile=profile)
#The following command will make sure that each engine is running in
# the right working directory to access the custom function(s).
client[:].map(os.chdir, [FILE_DIR]*len(client))
bview = client.load_balanced_view()



register_parallel_backend('ipyparallel',
                          lambda : IPythonParallelBackend(view=bview))

s = 'len(client)={}, joblib.cpu_count()={}, mpr.cpu_count()={}'.\
    format( len(client), joblib.cpu_count(), mpr.cpu_count())
print(s )

logging.basicConfig(filename=os.path.join(FILE_DIR,profile+'.log'),
                    filemode='w',
                    level=logging.DEBUG)
logging.info("number of CPUs found: {0}".format(cpu_count()))
logging.info("args.profile: {0}".format(profile))
logging.info("c.ids :{0}".format(str(client.ids)))
logging.info("{}".format(s ) )



from module_joblib_parallel_test import fun

with parallel_backend('ipyparallel'):
    ans = Parallel(n_jobs=len(client))(delayed(fun  )(i,i+i,i**2 )
                               for i in range(300 ) )

print('OUTPUT LEN {}'.format(len(ans) )  )
logging.info("{}".format(len(ans)  ) )
