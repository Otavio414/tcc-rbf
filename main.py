from RBF import RBF
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, max_error, r2_score, mean_squared_error

# Obtêm os dados de teste
dataframe_test = pd.read_csv(f'Dataset_Test.csv', delimiter=';', encoding='latin-1')
x_test = dataframe_test.iloc[:, 0:-1].values
y_test = dataframe_test.iloc[:, -1].values

# Obtêm os dados de treinamento
dataframe_train = pd.read_csv(f'Dataset_Train.csv', delimiter=';', encoding='latin-1')
x_train = dataframe_train.iloc[:, 0:-1].values
y_train = dataframe_train.iloc[:, -1].values

# Configura e treina a rede neural
network = RBF(n_neurons = 110)
network.fit(x_train, y_train)  

# Realiza os testes
y_pred = network.predict(x_pred=x_test)

# Apresenta o desempenho
print(f'{mean_squared_error(y_test, y_pred) * 100} | {mean_absolute_error(y_test, y_pred)*100} | {max_error(y_test, y_pred)*100} | {r2_score(y_test, y_pred)}')

# Exporta o resultado
results = pd.concat([pd.DataFrame(y_test*91.66296), pd.DataFrame(y_pred*91.66296)], axis=1)
results.columns = ['Real', 'Predição']
results.to_csv(f'Resultados.csv', index=False, encoding='latin-1')
