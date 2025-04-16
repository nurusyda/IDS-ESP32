import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_selection import SelectKBest, f_classif
import pickle

# Define paths to NSL-KDD dataset files (updated to root directory)
TRAIN_DATA_PATH = 'KDDTrain+.txt'
TEST_DATA_PATH = 'KDDTest+.txt'

# Column names for NSL-KDD dataset
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment',
    'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted',
    'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
]

def load_data(file_path):
    """Load NSL-KDD dataset from file."""
    print(f"Attempting to load: {file_path}")  # Debug print
    df = pd.read_csv(file_path, names=columns, header=None)
    return df

def preprocess_data(df):
    """Preprocess the dataset: encode categorical features, normalize numerical features."""
    # Separate features and labels
    X = df.drop(['label', 'difficulty'], axis=1)
    y = df['label']
    
    # Encode categorical features
    categorical_cols = ['protocol_type', 'service', 'flag']
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        encoders[col] = le
    
    # Convert labels to binary (normal: 0, anomaly: 1)
    y_binary = (y != 'normal').astype(int)
    
    # Select top 6 features using ANOVA F-test for efficiency
    selector = SelectKBest(score_func=f_classif, k=6)
    X_selected = selector.fit_transform(X, y_binary)
    selected_features = X.columns[selector.get_support()].tolist()
    
    # Normalize selected features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_selected)
    
    return X_scaled, y_binary, selected_features, scaler, encoders

def save_processed_data(X, y, selected_features, scaler, encoders, filename_prefix):
    """Save processed data and preprocessing objects."""
    np.save(f'{filename_prefix}_X.npy', X)
    np.save(f'{filename_prefix}_y.npy', y)
    with open(f'{filename_prefix}_features.pkl', 'wb') as f:
        pickle.dump(selected_features, f)
    with open(f'{filename_prefix}_scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open(f'{filename_prefix}_encoders.pkl', 'wb') as f:
        pickle.dump(encoders, f)

def main():
    # Load train and test datasets
    try:
        train_df = load_data(TRAIN_DATA_PATH)
        test_df = load_data(TEST_DATA_PATH)
    except FileNotFoundError as e:
        print(f"Error: {e}. Please ensure KDDTrain+.txt and KDDTest+.txt are in the project directory.")
        return
    
    # Preprocess datasets
    X_train, y_train, selected_features, scaler, encoders = preprocess_data(train_df)
    
    # Apply same scaler and features to test data
    X_test_full = test_df.drop(['label', 'difficulty'], axis=1)
    y_test = (test_df['label'] != 'normal').astype(int)
    for col in ['protocol_type', 'service', 'flag']:
        X_test_full[col] = encoders[col].transform(X_test_full[col])
    X_test_selected = X_test_full[selected_features]
    X_test = scaler.transform(X_test_selected)
    
    # Save processed data
    save_processed_data(X_train, y_train, selected_features, scaler, encoders, 'train')
    save_processed_data(X_test, y_test, selected_features, scaler, encoders, 'test')
    
    print(f"Selected features: {selected_features}")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

if __name__ == '__main__':
    main()