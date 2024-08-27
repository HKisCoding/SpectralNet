from scipy.io import loadmat 
import pandas as pd
data = loadmat(r'dataset/prokaryotic.mat')

label = data['truth'][:,0]



data  = {k:v for k, v in data.items() if k[0] != '_'}
df = pd.DataFrame({k: pd.Series(v[0]) for k, v in data.items()})  
df.to_csv('dataset/prokaryotic.csv', index=False)