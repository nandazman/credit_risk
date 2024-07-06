import joblib
from sklearn.model_selection import train_test_split
import pandas as pd

def load_data(fname):
    """
    Reads data from a CSV file.

    Parameters:
    fname (str): The file path of the CSV file to read.

    Returns:
    pandas.DataFrame: A DataFrame containing the data read from the CSV file.
    """
    csv_data = pd.read_csv(fname, sep = ",")
    print(f'Data shape               : {csv_data.shape}')

    return csv_data

def split_input_output(data, target_col):
    """
    Splits the input DataFrame into features (X) and target variable (y) based on
    the specified target column.

    Parameters:
    data (pandas.DataFrame): The input DataFrame containing features and target
    variable. target_col (str): The name of the column representing the target
    variable.

    Returns:
        - X (pandas.DataFrame): The DataFrame containing features
        (input variables) with the target column removed.
        - y (pandas.Series): The Series containing the target variable.
    """
    X = data.drop(target_col, axis = 1)
    y = data[target_col]

    print(f'Original data shape: {data.shape}')
    print(f'X data shape: {X.shape}')
    print(f'y data shape: {y.shape}')
    return X, y

def split_train_test(X, y, test_size, random_state = None):
    """
    Splits the input features and target variable into training and testing sets.

    Parameters:
    X (pandas.DataFrame): The DataFrame containing the input features.
    y (pandas.Series): The Series containing the target variable.
    test_size (float): The proportion of the dataset to include in the test split.
    random_state (int): The seed value for random state to ensure reproducibility.

    Returns:
    tuple: A tuple containing four elements:
        - X_train (pandas.DataFrame): The DataFrame containing the training input features.
        - X_test (pandas.DataFrame): The DataFrame containing the testing input features.
        - y_train (pandas.Series): The Series containing the training target variable.
        - y_test (pandas.Series): The Series containing the testing target variable.

    Splits the input features (X) and target variable (y) into training and testing sets.
    The `test_size` parameter specifies the proportion of the dataset to include in the test split.
    The `seed` parameter is used for random state to ensure reproducibility.
    Prints the shape of the training and testing input features and target variables.
    Returns a tuple containing the training and testing input features and target variables.
    """
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size = test_size,
                                                        stratify = y,
                                                        random_state = random_state)

    print(f'X train shape: f{X_train.shape}')
    print(f'X test shape : f{X_test.shape}')
    print(f'y train shape: f{y_train.shape}')
    print(f'y test shape : f{y_test.shape}\n')

    return  X_train, X_test, y_train, y_test

def serialize_data(data, path):
    """
    Serializes the given data to the specified path using joblib.

    Parameters:
    data (any): The instance of the object to be serialized.
    path (str): The file path where the serialized data will be stored.

    Returns:
    None
    """
    joblib.dump(data, path)

def deserialize_data(path):
    """
    Deserializes data from the specified path using joblib.

    Parameters:
    path (str): The file path from where the serialized data will be loaded.

    Returns:
    any: The deserialized data.
    """
    data = joblib.load(path)

    return data
