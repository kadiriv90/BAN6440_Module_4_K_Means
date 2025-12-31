"""
K-Means Clustering for IT Audit Monitoring & Fraud Detection

Dataset: CSE-CIC-IDS2018 (Processed Traffic Data)

DESIGN JUSTIFICATION:
For this academic study, we only use a representative subset of CSV files.
This keeps the computation efficient while preserving diversity of network traffic
patterns to demonstrate clustering effectively.

"""

# Import required libraries
import os                  # To work with file paths
import glob                # To find files matching patterns
import pandas as pd        # For data manipulation
import numpy as np         # For numerical operations
from sklearn.preprocessing import StandardScaler  # To standardize numeric data
from sklearn.cluster import KMeans                # K-Means clustering algorithm
from sklearn.decomposition import PCA            # For reducing dimensions (2D visualization)
import matplotlib.pyplot as plt                  # For plotting clusters

# ============================================================
# PATH CONFIGURATION
# ============================================================

# Get the absolute path to the project root dynamically
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to the folder containing processed CSV files
DATA_PATH = os.path.join(BASE_DIR, "data", "processed_traffic")

# Maximum number of files to load for this study (representative subset)
MAX_FILES_TO_LOAD = 2


# ============================================================
# DATA LOADING
# ============================================================

def load_data(folder_path: str, max_files: int) -> pd.DataFrame:
    """
    Load and combine a limited number of CSV files from the folder.

    Steps:
    1. Search for all CSV files in the folder.
    2. Select the first `max_files` files for processing.
    3. Read each CSV into a Pandas DataFrame.
    4. Combine all DataFrames into one.
    5. Print info about the loaded files and total records.

    :param folder_path: Absolute path to the CSV folder
    :param max_files: Number of files to load
    :return: Combined Pandas DataFrame
    """
    print(f"\nLooking for CSV files in:\n{folder_path}")

    # Find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))

    # If no CSV files found, raise an error
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in directory: {folder_path}")

    # Select only the first `max_files` for this study
    selected_files = csv_files[:max_files]

    print(f"\nUsing {len(selected_files)} CSV file(s):")
    for file in selected_files:
        print(f" - {file}")

    # Read each CSV into a DataFrame
    dataframes = [pd.read_csv(file, low_memory=False) for file in selected_files]

    # Combine all DataFrames into one
    combined_df = pd.concat(dataframes, ignore_index=True)

    print(f"\nTotal records loaded: {combined_df.shape[0]}")
    return combined_df


# ============================================================
# DATA PREPROCESSING
# ============================================================

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the data to make it ready for clustering.

    Steps:
    1. Convert all columns to numeric type. If a column contains text, it becomes NaN.
    2. Replace infinity values with NaN.
    3. Remove columns that are completely empty (all NaN).
    4. Replace remaining NaN with 0.
    5. Clip extremely large values to avoid numerical errors.
    6. Print number of numeric features retained and sample data.

    :param df: Raw DataFrame
    :return: Cleaned numeric DataFrame
    """
    # Convert all columns to numeric (non-numeric becomes NaN)
    numeric_df = df.apply(pd.to_numeric, errors="coerce")

    # Replace any positive/negative infinity with NaN
    numeric_df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Drop columns that are completely empty
    numeric_df.dropna(axis=1, how="all", inplace=True)

    # Replace remaining missing values with 0
    numeric_df.fillna(0, inplace=True)

    # Clip extremely large or small values to prevent overflow
    numeric_df = numeric_df.clip(-1e10, 1e10)

    # If no numeric columns remain, raise an error
    if numeric_df.shape[1] == 0:
        raise ValueError("No numeric features available after preprocessing.")

    # Print debug info
    print(f"Numeric features retained: {numeric_df.shape[1]}")
    print(f"Sample values:\n{numeric_df.iloc[:5, :5]}")  # First 5 rows, first 5 columns

    return numeric_df


def scale_data(df: pd.DataFrame) -> np.ndarray:
    """
    Standardize numeric features to have mean = 0 and std = 1.

    :param df: Clean numeric DataFrame
    :return: Scaled NumPy array
    """
    scaler = StandardScaler()  # Initialize the scaler
    return scaler.fit_transform(df)


# ============================================================
# K-MEANS CLUSTERING
# ============================================================

def perform_kmeans(data: np.ndarray, n_clusters: int = 4) -> np.ndarray:
    """
    Apply K-Means clustering to the scaled data.

    :param data: Scaled numeric data (NumPy array)
    :param n_clusters: Number of clusters to form
    :return: Array of cluster labels for each record
    """
    kmeans = KMeans(
        n_clusters=n_clusters,  # Number of clusters
        random_state=42,        # For reproducible results
        n_init=10               # Number of initializations for stability
    )
    return kmeans.fit_predict(data)


# ============================================================
# VISUALIZATION
# ============================================================

def visualize_clusters(data: np.ndarray, labels: np.ndarray) -> None:
    """
    Reduce data to 2D using PCA and plot clusters.

    :param data: Scaled numeric data
    :param labels: Cluster labels assigned by K-Means
    """
    pca = PCA(n_components=2)       # Reduce to 2 principal components
    reduced_data = pca.fit_transform(data)  # Transform data to 2D

    plt.figure(figsize=(10, 6))     # Create figure
    plt.scatter(
        reduced_data[:, 0],         # X-axis: first principal component
        reduced_data[:, 1],         # Y-axis: second principal component
        c=labels,                   # Color points by cluster label
        alpha=0.6                    # Make points slightly transparent
    )
    plt.title("K-Means Clustering of Network Traffic Data")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()


# ============================================================
# MAIN APPLICATION
# ============================================================

def main():
    """
    Main workflow of the application.
    """
    try:
        print("\nLoading representative subset of data...")
        df = load_data(DATA_PATH, MAX_FILES_TO_LOAD)

        print("\nPreprocessing data...")
        numeric_df = preprocess_data(df)

        if numeric_df.empty:
            raise ValueError("Preprocessed dataset is empty.")

        print("\nScaling features...")
        scaled_data = scale_data(numeric_df)

        print("Applying K-Means clustering...")
        cluster_labels = perform_kmeans(scaled_data, n_clusters=4)

        print("Visualizing clustering results...")
        visualize_clusters(scaled_data, cluster_labels)

        print("\nK-Means clustering completed successfully.")

    except Exception as error:
        # Catch any errors and print a clear message
        print(f"\nAPPLICATION ERROR: {error}")


# Run main workflow if this script is executed directly
if __name__ == "__main__":
    main()
