from pathlib import Path
import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    """
    Load dataset from a CSV file and clean column names.

    Args:
        file_path (str): Path to the dataset file.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If file does not exist.
        RuntimeError: If file cannot be read.
    """
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File not found at: {file_path}")

    try:
        df = pd.read_csv(path)

        # Clean column names (removes hidden spaces)
        df.columns = df.columns.str.strip()

    except Exception as e:
        raise RuntimeError(f"Error reading CSV file: {e}")

    return df