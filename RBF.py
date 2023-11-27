import numpy as np

class RBF:
    def __init__(self, n_neurons):
        self.__n_neurons = n_neurons
        self.__weights = np.zeros(n_neurons + 1)
        self.__centers = np.zeros(n_neurons)
        self.__std = 0

    @property
    def weights(self):
        return self.__weights

    @weights.setter
    def weights(self, weights):
        self.__weights = weights

    @property
    def centers(self):
        return self.__centers

    @centers.setter
    def centers(self, centers):
        self.__centers = centers

    @property
    def std(self):
        return self.__std

    @std.setter
    def std(self, std):
        self.__std = std

    # Função para calcular a função de base radial (RBF)
    def rbf_function(self, sample, center, std):
        return np.exp(-1 / (2 * std**2) * np.linalg.norm(sample - center) ** 2)

    # Função para treinar a rede RBF
    def fit(self, x_train, y_train):
        num_samples = len(x_train)
        # Inicialização dos centros das funções de base radial
        self.centers = np.array(x_train)[np.random.choice(num_samples, self.__n_neurons, replace=False)]
        # Calcula o desvio padrão para as funções de base radial
        self.std = np.std(x_train)
                     
       # Calcula as saídas das funções de base radial para todos os exemplos
        rbf_outputs = np.zeros((num_samples, self.__n_neurons))
        for i in range(num_samples):
            for j in range(self.__n_neurons):
                rbf_outputs[i, j] = self.rbf_function(x_train[i], self.centers[j], self.std)

         # Adiciona um termo de bias às saídas da RBF
        rbf_outputs = np.column_stack([rbf_outputs, np.ones(num_samples)])
        
        # Resolve o problema de regressão para obter os pesos da camada de saída
        self.weights = np.linalg.pinv(rbf_outputs).dot(y_train)
  
    # Função para realizar previsões
    def predict(self, x_pred):
        num_samples = len(x_pred)
        rbf_outputs = np.zeros((num_samples, self.__n_neurons))
        for i in range(num_samples):
            for j in range(self.__n_neurons):
                rbf_outputs[i, j] = self.rbf_function(x_pred[i], self.centers[j], self.std)
        
        # Adicione um bias às saídas da RBF
        rbf_outputs = np.column_stack([rbf_outputs, np.ones(num_samples)])
        
        # Faz as previsões
        predictions = rbf_outputs.dot(self.weights)
        
        return predictions