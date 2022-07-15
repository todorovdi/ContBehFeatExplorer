#!/usr/bin/bash
FMTSTR='%.11i %.2t %.12j %.7L %.5P'
# %D 
squeue -u todorov1 -o "$FMTSTR" | head "-$1"
#squeue -u todorov1 -o "$FMTSTR" | tail "-$1"
#
#default        "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R"
#-l, --long     "%.18i %.9P %.8j %.8u %.8T %.10M %.9l %.6D %R"
#-s, --steps    "%.15i %.8j %.9P %.8u %.9M %N"

#%i    Job or job step id.
#%t    Job  state  in  compact form.  See the JOB STATE CODES section below for a list of
#                    possible states.  (Valid for jobs only)
#%j    Job or job step name.  (Valid for jobs and job steps)
#%L    Time left for the job to execute in  days-hours:minutes:seconds. 
#%D    Number  of nodes allocated to the job or the minimum number of nodes required by a
#                    pending job.
  
