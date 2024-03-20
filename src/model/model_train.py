import logging
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

def train_rnn_with_cross_validation(model, X_train, y_train, skf):
    train_accuracies = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    val_f1_scores = []
    val_auc_rocs = []

    for i, (train_index, val_index) in enumerate(skf.split(X_train, y_train.argmax(1))):
        print(f'Fold {i+1}:')

        # Extract the train and validation folds
        x_train_fold, x_val_fold = X_train[train_index], X_train[val_index]
        y_train_fold, y_val_fold = y_train[train_index], y_train[val_index]

        # Reshape input data for RNN model 
        x_train_fold_reshaped = np.reshape(x_train_fold, (x_train_fold.shape[0], 1, x_train_fold.shape[1]))
        x_val_fold_reshaped = np.reshape(x_val_fold, (x_val_fold.shape[0], 1, x_val_fold.shape[1]))

        # Fit the RNN model
        history = model.fit(x_train_fold_reshaped, y_train_fold, validation_data=(x_val_fold_reshaped, y_val_fold), epochs=100, batch_size=32, verbose=1)

        # Evaluate train and validation accuracy
        train_accuracy = history.history['accuracy'][-1]
        val_accuracy = history.history['val_accuracy'][-1]
        print(f'Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

        # Append accuracies to lists
        train_accuracies.append(train_accuracy)
        val_accuracies.append(val_accuracy)

        # Log the training progress
        logging.info(f'Fold {i+1}: Train Accuracy: {train_accuracy}, Validation Accuracy: {val_accuracy}')

        # Predict on validation set for further evaluation
        y_pred_probs = model.predict(x_val_fold_reshaped)
        y_pred = (y_pred_probs > 0.5).astype(int)

        precision, recall, f1_score, _ = precision_recall_fscore_support(y_val_fold, y_pred, average='micro')
        auc_roc = roc_auc_score(y_val_fold, y_pred_probs, average='micro')

        val_precisions.append(precision)
        val_recalls.append(recall)
        val_f1_scores.append(f1_score)
        val_auc_rocs.append(auc_roc)

    return train_accuracies, val_accuracies, val_precisions, val_recalls, val_f1_scores, val_auc_rocs
