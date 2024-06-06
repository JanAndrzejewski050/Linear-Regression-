import pandas as pd
import numpy as np
from metrics import Metrics
from scalers import MinMaxScaler, StandardScaler


class Data_Prep:
    def __init__(self, file_name,test_size=0.2, random_state=None):
        self.data = pd.read_csv(file_name)
        self.n = len(self.data)
        self.m = len(self.data.columns)
        self.test_size = test_size
        self.random_state = random_state

    def clean_rows(self):
        # Filtering rows that have less then m elements
        self.data = self.data.dropna(thresh=self.m).reset_index(drop=True)
    
    def split(self):
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        shuffeld_indices = np.random.permutation(self.n)
        test_set_size = int(self.n * self.test_size)

        test_indices = shuffeld_indices[:test_set_size]
        train_indices = shuffeld_indices[test_set_size:]

        train_set = self.data.iloc[train_indices].reset_index(drop=True)
        test_set = self.data.iloc[test_indices].reset_index(drop=True)

        return train_set, test_set
    
    def split_data(self):
        self.train_set, self.test_set = self.split()
        self.x_train = self.train_set.iloc[:, :-1]
        self.y_train = self.train_set.iloc[:, -1]
        self.x_test = self.test_set.iloc[:, :-1]
        self.y_test = self.test_set.iloc[:, -1]

    def show_data(self):
        print(self.data)

    def show_train_test_sets(self):
        print("Train Set:"); print(self.train_set); print("Test Set:"); print(self.test_set)



class Data_Scale(Data_Prep):
    def __init__(self, file_name, scaling_type="minmax", test_size=0.2, random_state=None):
        super().__init__(file_name, test_size, random_state)
        self.scaling_type = scaling_type

        if scaling_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaling_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError("Unsupported scaling type")
    
    def fit_scaler(self):
        self.scaler.fit(self.x_train)

    def transform_data(self):
        x_train_scaled = self.scaler.transform(self.x_train)
        x_test_scaled = self.scaler.transform(self.x_test)
        return x_train_scaled, self.y_train, x_test_scaled, self.y_test




file_name = "data.csv"  # Podaj swoją ścieżkę do pliku CSV
scaling_type = "standard"  # Możesz użyć "minmax" lub "standard"

data_scale = Data_Scale(file_name, scaling_type)

# Czyszczenie danych
data_scale.clean_rows()

# Dzielimy dane na treningowe i testowe
data_scale.split_data()

# Dopasowujemy skaler do danych treningowych
data_scale.fit_scaler()

# Skalujemy dane i otrzymujemy wyniki
x_train_scaled, y_train, x_test_scaled, y_test = data_scale.transform_data()

print("Dane treningowe po skalowaniu:")
print(x_train_scaled)
print("Dane testowe po skalowaniu:")
print(x_test_scaled)
