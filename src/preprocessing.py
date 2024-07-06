import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def ohe_transform(dataset, subset, prefix, ohe):
    """
    Function to transform a dataset column using a pre-fitted OneHotEncoder.

    Parameters:
    dataset : pd.DataFrame
        The dataset to be transformed.
    subset : str
        The column name in the dataset to be transformed.
    prefix : str
        The prefix to be used for the new encoded columns.
    ohe : OneHotEncoder
        The pre-fitted OneHotEncoder instance.

    Returns:
    pd.DataFrame
        The dataset with the one-hot encoded columns appended.
    """
    if not isinstance(dataset, pd.DataFrame):
        raise RuntimeError("Fungsi ohe_transform: parameter dataset harus bertipe DataFrame!")
    
    if not isinstance(ohe, OneHotEncoder):
        raise RuntimeError("Fungsi ohe_transform: parameter ohe harus bertipe OneHotEncoder!")
    
    if not isinstance(prefix, str):
        raise RuntimeError("Fungsi ohe_transform: parameter prefix harus bertipe str!")
    
    if not isinstance(subset, str):
        raise RuntimeError("Fungsi ohe_transform: parameter subset harus bertipe str!")
    
    try:
        column_names = list(dataset.columns)
        column_names.index(subset)
    except ValueError:
        raise RuntimeError("Fungsi ohe_transform: parameter subset string namun data tidak ditemukan dalam daftar kolom yang terdapat pada parameter dataset.")
    
    print("Fungsi ohe_transform: parameter telah divalidasi.")

    dataset = dataset.copy()

    print(f"Fungsi ohe_transform: daftar nama kolom sebelum dilakukan pengkodean adalah {list(dataset.columns)}")

    col_names = [f"{prefix}_{col}" for col in ohe.categories_[0].tolist()]

    transformed_data = ohe.transform(dataset[[subset]]).toarray()
    encoded = pd.DataFrame(transformed_data, columns=col_names,index=dataset.index)
    dataset = pd.concat([dataset, encoded], axis=1)

    dataset.drop(columns=[subset], inplace=True)

    print(f'Fungsi ohe_transform: daftar nama kolom setelah dilakukan pengkodean adalah {list(dataset.columns)}')

    return dataset