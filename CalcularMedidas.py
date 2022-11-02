import math
import numpy as np
import pandas as pd
from sklearn import metrics

  
dfeli_ = []
dfreal_ = []
dfest_ = []

for i in range(6):
    print("mes " + str(i+1))
    dfeli_.append(np.asarray(pd.read_csv('DatosEliminados/Ventana_Eli_mes' + str(i+1) + '.csv'))[:,1:])
    dfreal_.append(np.asarray(pd.read_csv('DatosOriginales/Ventana_Ori_mes' + str(i+1) + '.csv'))[:,1:])
    dfest_.append(np.asarray(pd.read_csv('DatosEstimados/Ventana_Estimada_mes' + str(i+1) + '.csv')))


dfeli = np.concatenate(dfeli_, axis = 1)
dfreal = np.concatenate(dfreal_, axis = 1)
dfest = np.concatenate(dfest_, axis = 1)

mape = []
rmse = []
evs = []

for sensor in range(0,130): 
    mascara = (dfreal[sensor] - dfeli[sensor])>0
    try:
        mape.append(metrics.mean_absolute_percentage_error(dfreal[sensor][mascara],dfest[sensor][mascara]))
    except:
        mape.append(math.nan)
    try:
        rmse.append(metrics.mean_squared_error(dfreal[sensor][mascara],dfest[sensor][mascara], squared = False))
    except:
        rmse.append(math.nan)
    try:
        evs.append(metrics.explained_variance_score(dfreal[sensor][mascara],dfest[sensor][mascara]))
    except:
        evs.append(math.nan)

print('MAPE: ' + str(np.nanmean(mape)))
print('RMSE: ' + str(np.nanmean(rmse)))
print('EVS: ' + str(np.nanmean(evs)))
    


