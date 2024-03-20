import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Bidirectional, LSTM, Dense, BatchNormalization, Dropout
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import cross_val_score, cross_val_predict, KFold
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from keras.regularizers import l1, l2

class RNNModel:
    """This class creates bidirectional LSTM-based RNN model for classification 
    with dropout and softmax activation."""
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(None, self.num_input)),
            Dropout(0.1),
            Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001))),
            Dropout(0.1),
            LSTM(100, kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(self.num_output, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_model(self):
        return self.model
        

class LSTMModel:
    """This class creates LSTM-based RNN model for classification with dropout and 
    softmax activation."""
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            LSTM(100, return_sequences=True, input_shape=(None, self.num_input)),
            Dropout(0.1),
            LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            LSTM(100, kernel_regularizer=l2(0.001)),
            Dropout(0.1),
            Dense(self.num_output, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_model(self):
        return self.model

class BiRNNModel:
    """This class creates LSTM-based Bidirectional RNN model for classification with 
    batch normalization and softmax activation."""
    def __init__(self, num_input, num_output):
        self.num_input = num_input
        self.num_output = num_output
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential([
            Bidirectional(LSTM(100, return_sequences=True), input_shape=(None, self.num_input)),
            BatchNormalization(),
            Bidirectional(LSTM(100, return_sequences=True, kernel_regularizer=l2(0.001))),
            BatchNormalization(),
            LSTM(100, kernel_regularizer=l2(0.001)),
            BatchNormalization(),
            Dense(self.num_output, activation='softmax')
        ])

        model.compile(optimizer=Adam(0.0001),
                      loss='categorical_crossentropy',
                      metrics=['accuracy'])

        return model

    def get_model(self):
        return self.model
