#salloc --partition=batch --nodes=1 --account=icei-hbp-2020-0012 --time=00:50:00 --mem=200G
salloc --partition=batch --nodes=1 --account=icei-hbp-2020-0012 --time="00:"$1":00"  --mem=80G 
# json, pandas , scikit-learn, xgboost
# python -m sklearnex my_application.py 

#srun --cpu_bind=none --nodes=1 --pty /bin/bash -i
