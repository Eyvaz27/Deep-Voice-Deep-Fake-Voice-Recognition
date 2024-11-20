import numpy as np
import pandas as pd
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split

SEED = 42
DATA_PATH = "/workspaces/Deep-Voice-Deep-Fake-Voice-Recognition/dataset/deep_voice.csv"

def load_raw_csv(data_path):
    raw_data = pd.read_csv(filepath_or_buffer=data_path)
    target_map = lambda label: 1 if label=='FAKE' else 0
    raw_data.LABEL = raw_data.LABEL.map(arg=target_map, na_action='ignore')
    return raw_data

def sample_default(data_path=None):
    data_path = DATA_PATH if data_path is None else data_path
    raw_data = load_raw_csv(data_path)
    row = np.random.choice(np.arange(0, len(raw_data)))
    return raw_data.iloc[row, :-1]

def train_val_test(data_path, proportions):

    assert sum(proportions) == 1.0
    
    raw_data = load_raw_csv(data_path)
    feature_names = raw_data.columns[:-1]

    X = raw_data.loc[:, feature_names]
    y = raw_data["LABEL"]

    # Do train/validation/test split with 60%/20%/20% distribution.
    # Use the train_test_split function and set the random_state parameter to 1.
    test_p = proportions[2] 
    val_p = proportions[1]/proportions[0]
    X_full_train, X_test, y_full_train, y_test = train_test_split(X, y, test_size=test_p, 
                                                                random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_full_train, y_full_train, 
                                                    test_size=val_p, random_state=SEED)

    X_full_train.reset_index(drop=True, inplace=True)
    X_train.reset_index(drop=True, inplace=True)
    X_val.reset_index(drop=True, inplace=True)
    X_test.reset_index(drop=True, inplace=True)

    y_full_train.reset_index(drop=True, inplace=True)
    y_train.reset_index(drop=True, inplace=True)
    y_val.reset_index(drop=True, inplace=True)
    y_test.reset_index(drop=True, inplace=True)

    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

def preprocess_data(data_path=None, proportions=(0.6, 0.2, 0.2)):
    data_path = DATA_PATH if data_path is None else data_path
    train, val, test = train_val_test(data_path, proportions)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = train, val, test

    # # # following Note 1 in /analysis/feature_exploration.ipynb
    feature_transformer = RobustScaler()
    X_train = feature_transformer.fit_transform(X_train.values)
    X_val = feature_transformer.transform(X_val.values)
    X_test = feature_transformer.transform(X_test.values)
    return feature_transformer, (X_train, y_train.values), (X_val, y_val.values), (X_test, y_test.values)


