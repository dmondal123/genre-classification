import pytest
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from src.model.model_architecture import RNNModel, BiRNNModel, LSTMModel


X = np.load('X.npy')
y = np.load('y.npy')

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y, random_state=42)

# Test cases
def test_rnn_model():
    model = RNNModel(num_input=X.shape[1], num_output=y.shape[1])
    assert isinstance(model, RNNModel)
    assert model.num_input == X_train.shape[1]
    assert model.num_output == y_train.shape[1]

def test_birnn_model():
    model = BiRNNModel(num_input=X.shape[1], num_output=y.shape[1])
    assert isinstance(model, BiRNNModel)
    assert model.num_input == X_train.shape[1]
    assert model.num_output == y_train.shape[1]

def test_lstm_model():
    model = LSTMModel(num_input=X.shape[1], num_output=y.shape[1])
    assert isinstance(model, LSTMModel)
    assert model.num_input == X_train.shape[1]
    assert model.num_output == y_train.shape[1]

def test_data_splitting():
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(y_train) > 0
    assert len(y_test) > 0

def test_cross_validation():
    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train.argmax(1))):
        x_train_fold, x_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]
        assert len(x_train_fold) > 0
        assert len(x_val_fold) > 0
        assert len(y_train_fold) > 0
        assert len(y_val_fold) > 0


if __name__ == "__main__":
    pytest.main()