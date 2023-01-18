#salloc --partition=batch --nodes=1 --account=icei-hbp-2020-0012 --time=00:50:00
salloc --partition=batch --nodes=1 --account=icei-hbp-2020-0012 --time="00:"$1":00"
# json, pandas , scikit-learn, xgboost
# python -m sklearnex my_application.py 

