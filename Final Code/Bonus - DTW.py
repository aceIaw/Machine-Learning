import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys

display_distances = False

run_user_independent_cv = False

run_user_dependent_cv = False

#Read the CSV file
df = pd.read_csv()

# Extract the user, gesture, and sequence columns
def extract_sequences(df): # Focus on sequences by placing them in a dictionary and grouping them by user and gesture
    sequences = defaultdict(list)  # Initiate a dictionary that will hold sequences for each user and gesture
    for (user, gesture, sequence_id), group in df.groupby(['user', 'gesture', 'sequence']): # Groups rows by user, gesture, and sequence
        sequence_data = group[['x', 'y', 'z']].to_numpy() # Converts the x, y, z columns to a numpy array to facilitate calculations for the DTW algorithm
        sequence_data -= np.mean(sequence_data, axis=0) # Normalizes the sequence data by subtracting the mean
        sequences[(user, gesture)].append(sequence_data) # Appends the normalized sequence data to the dictionary under the corresponding user and gesture
    return sequences


# Calculating the barycenter of each gesture
def compute_gesture_barycenters(sequences): # Computes the barycenter of each gesture by averaging all its sequences, resulting in a "average gesture"
    gesture_groups = defaultdict(list) # Initiate a new dictionary where it is necessary to group sequences by gesture because we will compute the barycenter for each gesture
    for (user, gesture), seq_list in sequences.items(): # For a given user and gesture, we will extract the list of sequences
        gesture_groups[gesture].extend(seq_list) # Adds matching sequences to the corresponding gesture group
    gesture_barycenters = {} # Initiate a new dictionary to hold the barycenters for each gesture
    for gesture, seqs in gesture_groups.items(): # Gets all sequences for a given gesture
        min_len = min(seq.shape[0] for seq in seqs) # Finds the minimum length of the sequences of a specific gesture, with seq.shape[0] determining the number of rows in the sequence and thus taking the length of the shortest sequence
        aligned_seqs = [seq[:min_len] for seq in seqs] # Align all sequences to the minimum length sequence of a specific gesture
        stacked = np.stack(aligned_seqs)  # Regroups all now sequences changed sequences of the same gesture into a 3D array
        barycenter = np.mean(stacked, axis=0)  # Calculates the mean of the sequences for every point in time for a given gesture, resulting in a "average gesture"
        gesture_barycenters[gesture] = barycenter # Adds the barycenter to the dictionary under the corresponding gesture
    return gesture_barycenters

# DTW distance function
def dtw_distance(seq1, seq2): # Builds a DTW matrix by calculating the distance between two sequences
    n, m = len(seq1), len(seq2) # Gets the lengths of two sequences, therefore determining of how long a given gesture is performed
    dtw_matrix = np.full((n + 1, m + 1), np.inf) # Set up the DTW matrix where the +1 is added to the lengths of the sequences to ensure to set up the first iteration correctly, annd np.inf is used to initialize the matrix with infinite values
    dtw_matrix[0, 0] = 0 # Sets up our starting point as 0, since we haven't calculated the distance yet
    for i in range(1, n + 1): # Loops through seq1, we start at 1 because our matrix cost at 0,0 is already set to 0
        for j in range(1, m + 1): # Loops through seq2
            cost = np.linalg.norm(seq1[i - 1] - seq2[j - 1])  # Calculates the euclidean distance between the two points in the sequences, the i-1 and j-1 are set to use the first index
            dtw_matrix[i, j] = cost + min( # Calculates the DTW Matrix by adding the minimum of the three previous costs (left, top, and left upper diagonal) to the current cost
                dtw_matrix[i - 1, j],
                dtw_matrix[i, j - 1],  
                dtw_matrix[i - 1, j - 1])

    return dtw_matrix[n, m]


# Helps visualise the distances between gestures
def print_pairwise_dtw_distances(gesture_barycenters):
    gestures = sorted(gesture_barycenters.keys())
    for i in range(len(gestures)):
        for j in range(i + 1, len(gestures)):
            g1, g2 = gestures[i], gestures[j]
            seq1 = gesture_barycenters[g1]
            seq2 = gesture_barycenters[g2]
            dist = dtw_distance(seq1, seq2)
            print(f"distance between {g1} {g2} = {round(dist, 2)}")
            sys.stdout.flush()

sequences = extract_sequences(df)
gesture_barycenters = compute_gesture_barycenters(sequences)
    
if display_distances:
    print_pairwise_dtw_distances(gesture_barycenters)


# User-independent cross-validation
def evaluate_user_independent_with_conf_matrix(sequences, k=1): # We compute k = 1 to just compare the sequence to the nearest neighbour
    users = sorted(set(u for u, _ in sequences.keys())) # Gets a list of all users sorted alphabetically
    accuracies = [] # Will hold the accuracies for each user
    all_y_true, all_y_pred = [], [] # Will hold the true and predicted labels

    for test_user in tqdm(users, desc="User-Independent"): # Chooses one user at a time to test, tqdm is used to show the progress bar
        train = {key: value for key, value in sequences.items() if key[0] != test_user} # Adds sequences into a new dictionary of all users except the one we are testing
        test = {key: value for key, value in sequences.items() if key[0] == test_user} # Adds sequences into a new dictionary of the user we are testing

        y_true, y_pred = [], [] # Will hold the true and predicted labels for the current user

        for (user, true_gesture), gesture_sequences in test.items(): # Goes through all the sequences of the user we are testing
            for test_seq in gesture_sequences: # Runs through the numpy array a sequence
                distances = [] # Will hold the distances between the test sequence and all the training sequences
                for (train_user, train_gesture), train_sequences in train.items(): # Loops over every user, gesture, and sequence except the one we are testing
                    for train_seq in train_sequences: # Runs through the numpy array a sequence
                        dist_dtw = dtw_distance(test_seq, train_seq) # Computes the DTW distance between the test sequence and the training sequence
                        distances.append((dist_dtw, train_gesture)) # Adds the distance and the associated gesture to the list
                distances.sort() # Sorts the distances in ascending order
                nearest = [g for _, g in distances[:k]] # Extracts the gesture from the distance list that are the nearest neighbours
                predicted = max(set(nearest), key=nearest.count) # In case k > 1, we use this helps to determine the nearest neighbour
                y_true.append(true_gesture) # Adds the current true gesture to the list
                y_pred.append(predicted) # Adds the predicted gesture to the list

        acc = accuracy_score(y_true, y_pred) # Computes the accuracy score by comparing the true and predicted labels
        accuracies.append(acc) # Adds the calculated accuracy to a list
        all_y_true.extend(y_true) # Adds the true labels to a list
        all_y_pred.extend(y_pred) # Adds the predicted labels to a list

    print("User-Independent Accuracy: {:.2f}% ± {:.2f}".format(
        100*np.mean(accuracies), 100*np.std(accuracies))) #Calculates the mean and sd of the total accuracies

    labels = sorted(set(all_y_true + all_y_pred)) # Remove duplicates from the true and predicted labels and sort them
    confusion = confusion_matrix(all_y_true, all_y_pred, labels=labels) # Determine the axis and labels of the confusion matrix
    display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
    display.plot(xticks_rotation=45, cmap="Blues")
    plt.title("DTW Confusion Matrix User-Independent")
    plt.show()

    return accuracies

sequences = extract_sequences(df)
if run_user_independent_cv:
    evaluate_user_independent_with_conf_matrix(sequences, k=1)

# User-dependent cross-validation
def evaluate_leave_one_sample_per_gesture_out(sequences, k=1):
    users = sorted(set(u for u, _ in sequences.keys())) # Gets a list of all users sorted alphabetically
    accuracies = [] # Will hold the accuracies for each user
    all_y_true, all_y_pred = [], [] # Will hold the true and predicted labels

    for user in tqdm(users, desc="Leave-One-Sample-Per-Gesture-Out CV"): # Chooses one user at a time to test, tqdm is used to show the progress bar
        user_data = {key: value for key, value in sequences.items() if key[0] == user} # Adds sequences into a new dictionary of the user we are testing
        for i in range(10): # For each gesture, we will leave one sample out at a time
            train_data = [] # Will hold the training data
            test_data = [] # Will hold the test data
            for (u, gesture), seq_list in user_data.items(): # Loops over every user's gesture and sequence
                test_seq = seq_list[i] # Put the ith sequence as the test sequence
                train_seqs = [seq for j, seq in enumerate(seq_list) if j != i] # Build a training sequence by taking all the sequences except for our test sequence
                test_data.append((gesture, test_seq)) # Adds the test sequence to the test data
                train_data.extend([(gesture, seq) for seq in train_seqs]) # Adds the multiple training sequences to the training data

            X_train = [seq for _, seq in train_data] # Contains a list of all the training sequences as numpy arrays
            y_train = [g for g, _ in train_data] # Contains a list of the corresponding gestures for the training sequences

            y_true, y_pred = [], [] # Will hold the true and predicted labels for the current user, and rests for the next gesture

            for true_gesture, test_seq in test_data: # Loops over the test data
                distances = [] # Stores the distance between the test sequence and all the training sequences
                for train_seq, train_gesture in zip(X_train, y_train): # Loops over the training sequences and their corresponding gestures, zip is used to combine the two lists
                    dist_dtw = dtw_distance(test_seq, train_seq) # Computes the DTW distance between the test sequence and the training sequence
                    distances.append((dist_dtw, train_gesture)) # Adds the dtw distance and the associated gesture label to the distances list
                distances.sort() # Sorts the distances in ascending order
                nearest = [g for _, g in distances[:k]] # Extracts the gesture from the distance list that are the nearest neighbours
                predicted = max(set(nearest), key=nearest.count) # In case k > 1, we use this helps to determine the nearest neighbour
                y_true.append(true_gesture) # Adds the current true gesture to the list
                y_pred.append(predicted) # Adds the predicted gesture to the list

            acc = accuracy_score(y_true, y_pred) # Computes the accuracy score by comparing the true and predicted labels
            accuracies.append(acc) # Adds the calculated accuracy to a list
            all_y_true.extend(y_true) # Adds the true labels to a list
            all_y_pred.extend(y_pred) # Adds the predicted labels to a list

    print("Leave-One-Gesture-Sample-Out Accuracy: {:.2f}% ± {:.2f}".format(
        100 * np.mean(accuracies), 100 * np.std(accuracies))) #Calculates the mean and sd of the total accuracies

    labels = sorted(set(all_y_true + all_y_pred)) # Remove duplicates from the true and predicted labels and sort them
    cm = confusion_matrix(all_y_true, all_y_pred, labels=labels) # Determine the axis and labels of the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(xticks_rotation=45, cmap="Oranges")
    plt.title("DTW Confusion Matrix (Leave-One-Gesture-Sample-Out CV)")
    plt.show()

    return accuracies

sequences = extract_sequences(df)

if run_user_dependent_cv:
    evaluate_leave_one_sample_per_gesture_out(sequences, k=1)
