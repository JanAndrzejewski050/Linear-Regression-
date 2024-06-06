import pandas as pd
import numpy as np
from metrics import Metrics


class Data_Prep:
    def __init__(self, file_name,test_size=0.2, random_state=None):
        self.data = pd.read_csv(file_name)
        self.n = len(self.data)
        self.m = len(self.data.columns)
        self.test_size = test_size
        self.random_state = random_state
        self.train_set, self.test_set = self.split()

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

    def show_data(self):
        print(self.data)

    def show_train_test_sets(self):
        print("Train Set:"); print(self.train_set); print("Test Set:"); print(self.test_set)


class Scalers(Metrics):
    @staticmethod
    def standar_scaler():
        pass

    @staticmethod
    def minmax_normalization():
        pass


# class Data_Scale(Data_Prep):
#     def __init__(self, file_name, test_size=0.2, random_state=None):
#         super().__init__(file_name, test_size, random_state)



d = Data_Prep('data.csv')
d.clean_rows()
d.show_data()
d.show_train_test_sets()
