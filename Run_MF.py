import sys
import numpy as np
import pandas as pd
import MF

for i in range(6):
    
    df = pd.read_csv('DatosEliminados/Ventana_Eli_mes'+ str(i+1) +'.csv', sep=',', index_col=0)
    Matriz_Eva = np.asarray(df)
    MetodoMF = MF.MF(R = Matriz_Eva, K = 360, beta = 0.1, iterations = 100) 
    MetodoMF.train()
    ME = MetodoMF.full_matrix()
    completado = pd.DataFrame(ME)
    completado.to_csv('DatosEstimados/Ventana_Estimada_mes' + str(i+1) + '.csv', index = False)

