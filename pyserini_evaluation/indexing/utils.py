import os, shutil
from scipy.sparse import csr_matrix

def maybe_create_folder(path):
    """
    Creates a directory if it does not exist.

    Parameters:
    path (str): The path of the directory to be created.
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
            print(f"Directory '{path}' created successfully!")
        else:
            print(f"Directory '{path}' already exists.")
    except OSError as e:
        print(f"Error creating directory '{path}': {e}")


def remove_folder(folder_path):
    """
    Removes a folder and all its contents.
    
    Args:
    folder_path (str): The path to the folder to be removed.
    
    Returns:
    bool: True if the folder was successfully removed, False otherwise.
    """
    try:
        # Check if the folder exists
        if os.path.exists(folder_path):
            # Remove the folder and all its contents
            shutil.rmtree(folder_path)
            print(f"Folder '{folder_path}' has been successfully removed.")
            return True
        else:
            print(f"The folder '{folder_path}' does not exist.")
            return False
    except Exception as e:
        print(f"An error occurred while trying to remove the folder: {e}")
        return False
    

def torch_csr_to_scipy_csr(torch_csr):
    """
    Convert a PyTorch sparse CSR matrix to a SciPy sparse CSR matrix.

    Parameters:
        torch_csr (torch.Tensor): PyTorch sparse CSR matrix.

    Returns:
        scipy.sparse.csr_matrix: Equivalent SciPy sparse CSR matrix.
    """
    crow_indices = torch_csr.crow_indices().numpy()
    col_indices = torch_csr.col_indices().numpy()
    values = torch_csr.values().numpy()

    scipy_csr = csr_matrix((values, col_indices, crow_indices), shape = torch_csr.shape)
    return scipy_csr


def merge_dicts(dict1, dict2):
    """
    Merges two dictionaries. If both dictionaries have the same key, the higher value is kept.
    
    Args:
        dict1 (dict): The first dictionary.
        dict2 (dict): The second dictionary.

    Returns:
        dict: A merged dictionary with the higher value for duplicate keys.
    """
    merged_dict = dict1.copy()  # Start with a copy of the first dictionary
    for key, value in dict2.items():
        if key in merged_dict:
            merged_dict[key] = max(merged_dict[key], value)  # Keep the higher value
        else:
            merged_dict[key] = value  # Add new key-value pair
    return merged_dict