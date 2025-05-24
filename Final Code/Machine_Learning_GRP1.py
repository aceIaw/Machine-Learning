import pandas as pd
import zipfile
import tempfile
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
from collections import defaultdict
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, Masking, Bidirectional, Input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
import os, joblib, time 





def MergeData():


    # === PARAMETERS ===
    zip_path = r'Gestures.zip'  # Update this as needed

    # Get the folder where the script is located
    script_dir = os.path.dirname(os.path.abspath(_file_))
    output_csv_path = os.path.join(script_dir, 'RAW_DATA.csv')

    # === UNZIP TO TEMPORARY DIRECTORY ===
    with tempfile.TemporaryDirectory() as temp_dir:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(temp_dir)

        # === PROCESS CSV FILES ===
        RAW_DATA = []
        for root, _, files in os.walk(temp_dir):
            for file in files:
                if file.endswith('.csv') and file.startswith('Subject'):
                    try:
                        parts = file.rstrip('.csv').split('-')
                        i2 = int(parts[0].replace('Subject', ''))
                        i3 = int(parts[1])
                        i4 = int(parts[2])
                    except Exception as e:
                        print(f"Filename format error in '{file}': {e}")
                        continue

                    file_path = os.path.join(root, file)
                    try:
                        df = pd.read_csv(file_path)
                        x_col = df['<x>'].tolist()
                        y_col = df['<y>'].tolist()
                        z_col = df['<z>'].tolist()
                        t_col = df['<t>'].tolist()

                        for i in range(len(x_col)):
                            RAW_DATA.append([i2, i3, i4, x_col[i], y_col[i], z_col[i], t_col[i]])
                    except Exception as e:
                        print(f"Error processing file '{file_path}': {e}")

    # === CONVERT TO DATAFRAME AND EXPORT WITHOUT HEADER ===
    df = pd.DataFrame(RAW_DATA)
    df.to_csv(output_csv_path, index=False, header=False)

    print("Le fichier CSV a été créé avec succès dans le dossier du script.")
MergeData() if not os.path.exists("RAW_DATA.csv") else print("RAW_DATA.csv already exists")  # Create the file only if it’s missing, otherwise do nothing


standardize = False 
normalize = False

def TransformData():


    file_path = r"RAW_DATA.csv" # Path to the raw data file
    df = pd.read_csv(file_path, header=None) # Read the CSV file without headers

    df.columns = ['user', 'gesture', 'sequence', 'x', 'y', 'z', 'time'] # Assign column names

    sensor_columns = ['x', 'y', 'z'] # Columns to be processed

    # If you only want your raw data, keep false for both


    for col in sensor_columns: # Iterate over sensor columns

        if standardize: # Standardization
            std = df[col].std() # Calculate standard deviation
            df[col] = (df[col] - df[col].mean()) / std # Standardize the column

        elif normalize: # Normalization
            min_val = df[col].min() # Calculate minimum value
            max_val = df[col].max() # Calculate maximum value
            df[col] = (df[col] - min_val) / (max_val - min_val) # Normalize the column


        output_suffix = "standardize" if standardize else "normalize" if normalize else "raw_data_with_columns" # Suffix for the output file
        output_path = fr"{output_suffix}.csv" # Path to save the processed data
        df.to_csv(output_path, index=False)

        # Print statistics
        print("\nProcessed Data Statistics:")
        print("Means:\n", df[sensor_columns].mean())
        print("Standard Deviations:\n", df[sensor_columns].std())
        print("Min values:\n", df[sensor_columns].min())
        print("Max values:\n", df[sensor_columns].max())
TransformData()

run_user_independent_DTW= False
run_user_dependent_DTW = False

def DTW():
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


    # User-independent cross-validation
    def user_independent(sequences, k=1): # We compute k = 1 to just compare the sequence to the nearest neighbour
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

        print("DTW User-Independent Accuracy: {:.2f}% ± {:.2f}".format(
            100*np.mean(accuracies), 100*np.std(accuracies))) #Calculates the mean and sd of the total accuracies

        labels = sorted(set(all_y_true + all_y_pred)) # Remove duplicates from the true and predicted labels and sort them
        confusion = confusion_matrix(all_y_true, all_y_pred, labels=labels) # Determine the axis and labels of the confusion matrix
        display = ConfusionMatrixDisplay(confusion_matrix=confusion, display_labels=labels)
        display.plot(xticks_rotation=45, cmap="Blues")

        plt.xlabel("Predicted Gesture")
        plt.ylabel("True Gesture")
        plt.title("DTW User-Independent Confusion Matrix")
        plt.show()

        return accuracies

    sequences = extract_sequences(df)
    if run_user_independent_DTW:
        user_independent(sequences, k=1)

    # User-dependent cross-validation
    def user_dependent(sequences, k=1):
        users = sorted(set(u for u, _ in sequences.keys())) # Gets a list of all users sorted alphabetically
        accuracies = [] # Will hold the accuracies for each user
        all_y_true, all_y_pred = [], [] # Will hold the true and predicted labels

        for user in tqdm(users, desc="User-Dependent"): # Chooses one user at a time to test, tqdm is used to show the progress bar
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

        print("DTW User-Dependent Accuracy: {:.2f}% ± {:.2f}".format(
            100 * np.mean(accuracies), 100 * np.std(accuracies))) #Calculates the mean and sd of the total accuracies

        labels = sorted(set(all_y_true + all_y_pred)) # Remove duplicates from the true and predicted labels and sort them
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels) # Determine the axis and labels of the confusion matrix
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(xticks_rotation=45, cmap="Oranges")

        plt.xlabel("Predicted Gesture")
        plt.ylabel("True Gesture")
        plt.title("DTW User-Dependent Confusion Matrix")
        plt.show()

        return accuracies

    sequences = extract_sequences(df)

    if run_user_dependent_DTW:
        user_dependent(sequences, k=1)
DTW()

run_user_independent_LSTM= False
run_user_dependent_LSTM = False

def LSTM():
    CACHE_DIR = "lstm_knn_cache_userind"
    os.makedirs(CACHE_DIR, exist_ok=True)

    # 1. Charger & grouper (user‐independent: on regroupe par user + gesture)
    df = pd.read_csv("Standardize.csv")
    df = df[df['x'] != 'x']
    df[['x','y','z','time']] = df[['x','y','z','time']].astype(float)
    df['gesture'] = df['gesture'].astype(int)

    # sequences_user[user] = list of (gesture_id, np.array of shape (T,4))
    sequences_user = defaultdict(list)
    grouped = df.groupby(['user','gesture','sequence'])
    for (user, gesture, _), grp in grouped:
        seq = grp.sort_values('time')[['x','y','z','time']].to_numpy()
        sequences_user[user].append((gesture, seq))

    # 2. Calculer global_max (une seule fois pour toutes les séquences)
    all_seqs = [seq for seqs in sequences_user.values() for _, seq in seqs]
    global_max = max(seq.shape[0] for seq in all_seqs)
    def user_independent():
        # 3. Fonction de prétraitement (interpolation + scaling)
        def preprocess(sequences, max_len=global_max, scaler=None, fit_scaler=True):
            X, y = [], []
            for gesture_id, seq in sequences:
                ori = seq.shape[0]
                oi = np.linspace(0,1,ori)
                ti = np.linspace(0,1,max_len)
                feats = [np.interp(ti, oi, seq[:,c]) for c in range(4)]
                X.append(np.stack(feats,axis=1))
                y.append(gesture_id)
            X = np.array(X)
            if fit_scaler:
                scaler = StandardScaler()
                X = scaler.fit_transform(X.reshape(-1,4)).reshape(X.shape)
            else:
                X = scaler.transform(X.reshape(-1,4)).reshape(X.shape)
            y = to_categorical(y, num_classes=10)
            return X, y, scaler

        # 4. Construction et compilation du modèle LSTM (une seule fois)
        def build_lstm(input_shape=(global_max,4)):
            inp = Input(shape=input_shape)
            x = Masking(0.)(inp)
            x = LSTM(128, return_sequences=True)(x)
            x = Dropout(0.1)(x)
            x = Bidirectional(LSTM(64))(x)
            x = Dropout(0.1)(x)
            x = Dense(64, activation='relu')(x)
            emb = Dense(32, activation='relu', name="embedding")(x)
            out = Dense(10, activation='softmax')(emb)
            m = Model(inputs=inp, outputs=out)
            m.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
            return m

        model = build_lstm()  # ne sera jamais reconstruit

        # 5. Leave-one-user-out (user-independent) avec cache et barre de progression
        all_y_true, all_y_pred = [], []
        per_user_accuracies = []
        users = sorted(sequences_user.keys())
        start = time.time()

        for u in tqdm(users, desc="User-independent eval", unit="user"):
            # 5.a. Préparer train/test pour utilisateur u
            train_seqs = [item for usr, seqs in sequences_user.items() if usr != u for item in seqs]
            test_seqs  = sequences_user[u]

            # 5.b. Prétraitement
            Xtr, ytr, scaler = preprocess(train_seqs, fit_scaler=True)
            Xte, yte, _      = preprocess(test_seqs,  scaler=scaler, fit_scaler=False)

            # 5.c. Définir chemins cache (poids en .weights.h5)
            wp = f"{CACHE_DIR}/lstm_weights_user{u}.weights.h5"
            sp = f"{CACHE_DIR}/scaler_user{u}.joblib"
            xp = f"{CACHE_DIR}/Xemb_user{u}.npy"
            yp = f"{CACHE_DIR}/Yemb_user{u}.npy"

            # 5.d. Charger ou entraîner + sauvegarder
            if os.path.exists(wp) and os.path.exists(sp) and os.path.exists(xp) and os.path.exists(yp):
                model.load_weights(wp)
                scaler = joblib.load(sp)
                Xtr_emb = np.load(xp)
                ytr_lab = np.load(yp)
            else:
                model.fit(Xtr, ytr, validation_split=0.1, epochs=100, batch_size=16, verbose=0)
                model.save_weights(wp)
                joblib.dump(scaler, sp)
                emb_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
                Xtr_emb = emb_model.predict(Xtr, verbose=0)
                ytr_lab = np.argmax(ytr, axis=1)
                np.save(xp, Xtr_emb)
                np.save(yp, ytr_lab)

            # 5.e. Extraire embeddings test + k-NN
            emb_model = Model(inputs=model.input, outputs=model.get_layer("embedding").output)
            Xte_emb = emb_model.predict(Xte, verbose=0)
            knn = KNeighborsClassifier(n_neighbors=5)
            knn.fit(Xtr_emb, ytr_lab)
            y_pred = knn.predict(Xte_emb)

            y_true = np.argmax(yte, axis=1)
            all_y_true.extend(y_true)
            all_y_pred.extend(y_pred)
            per_user_accuracies.append(accuracy_score(y_true, y_pred))

        # 6. Affichage du résultat final
        mean_acc = np.mean(per_user_accuracies)
        std_acc = np.std(per_user_accuracies)

        print(f"\nUser-independent Accuracy: {(mean_acc * 100):.2f}% (±{(std_acc * 100):.2f}%)")

        cm = confusion_matrix(all_y_true, all_y_pred, labels=range(10))
        cm_pct = cm.astype(float) / cm.sum(axis=1)[:, None] * 100

        disp = ConfusionMatrixDisplay(confusion_matrix=np.round(cm_pct), display_labels=range(10))
        disp.plot(cmap="Blues", values_format=".0f")
        plt.xlabel("Predicted Gesture")
        plt.ylabel("True Gesture")
        plt.title("LSTM User-Independent Confusion Matrix")
        plt.show()

        print(f"Durée totale : {(time.time() - start) / 60:.1f} minutes")
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



    if run_user_dependent_LSTM :
        user_dependent(all_seqs)
    if run_user_independent_LSTM :
        user_independent(all_seqs)
LSTM()




        