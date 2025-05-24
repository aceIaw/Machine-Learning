import pandas as pd
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm




run_LSTM_user_dependent = True



# User-dependent cross-validation
def user_dependent(sequences, k=1):
    users = sorted(set(u for u, _ in sequences.keys())) # Gets a list of all users sorted alphabetically
    accuracies = [] # Will hold the accuracies for each user
    all_y_true, all_y_pred = [], [] # Will hold the true and predicted labels

    for user in tqdm(users, desc="User-Dependent"): # Chooses one user at a time to test, tqdm is used to show the progress bar
        user_data = {key: value for key, value in sequences.items() if key[0] == user} # Adds sequences into a new dictionary of the user we are testing
        for i in range(10):  # For each gesture, we will leave one sample out at a time
            train_data = [] # Will hold the training data
            test_data = [] # Will hold the test data
            for (u, gesture), seq_list in user_data.items(): # Loops over every user's gesture and sequence
                test_seq = seq_list[i] # Put the ith sequence as the test sequence
                train_seqs = [seq for j, seq in enumerate(seq_list) if j != i] # Build a training sequence by taking all the sequences except for our test sequence
                test_data.append((gesture, test_seq)) # Adds the test sequence to the test data
                train_data.extend([(gesture, seq) for seq in train_seqs]) # Adds the multiple training sequences to the training data


            max_len = max(seq.shape[0] for _, seq in train_data + test_data) # Gets the maximum length of the sequences


            grouped_train = [(g, [s]) for g, s in train_data] # Groups the training data by gesture
            grouped_test = [(g, [s]) for g, s in test_data] # Groups the test data by gesture

            X_train, y_train, scaler = preprocess_sequences(grouped_train, max_len=max_len, fit_scaler=True) # Preprocess the training data
            X_test, y_test, _ = preprocess_sequences(grouped_test, max_len=max_len, scaler=scaler, fit_scaler=False) # Preprocess the test data

            model = build_full_lstm((max_len, 4)) # Build the LSTM model
            model.fit(X_train, y_train, validation_split=0.1, epochs=100, batch_size=16, verbose=0) # Train the model

            embedding_model = extract_embedding_model(model) # Extract the embedding model
            X_train_embed = embedding_model.predict(X_train, verbose=0) # Get the embeddings for the training data
            X_test_embed = embedding_model.predict(X_test, verbose=0) # Get the embeddings for the test data

            y_train_labels = np.argmax(y_train, axis=1) # Get the labels for the training data
            y_test_labels = np.argmax(y_test, axis=1) # Get the labels for the test data

            knn = KNeighborsClassifier(n_neighbors=k) # Create the k-NN classifier
            knn.fit(X_train_embed, y_train_labels) # Fit the classifier to the training data
            y_pred = knn.predict(X_test_embed) # Predict the labels for the test data

            acc = accuracy_score(y_test_labels, y_pred) # Computes the accuracy score by comparing the true and predicted labels
            accuracies.append(acc) # Adds the calculated accuracy to a list
            all_y_true.extend(y_test_labels) # Adds the true labels to a list
            all_y_pred.extend(y_pred) # Adds the predicted labels to a list


    print("LSTM User-Dependent Accuracy: {:.2f}% ± {:.2f}".format(
        100 * np.mean(accuracies), 100 * np.std(accuracies))) #Calculates the mean and sd of the total accuracies


    labels = sorted(set(all_y_true + all_y_pred)) # Remove duplicates from the true and predicted labels and sort them
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels) # Determine the axis and labels of the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45, cmap="Oranges")

    plt.xlabel("Predicted Gesture")
    plt.ylabel("True Gesture")
    plt.title("LSTM User-Dependent")
    plt.show()

    return accuracies



if run_user_dependent:
    user_dependent(sequences)
