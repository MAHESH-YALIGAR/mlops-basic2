
import pandas as pd
import os
import yaml
import logging

from sklearn.model_selection import train_test_split


# --------------------------------------------------
# 1. Logging configuration
# --------------------------------------------------

log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel(logging.DEBUG)

# Prevent duplicate log messages if the module is reloaded
logger.propagate = False

if not logger.handlers:
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)

    log_file_path = os.path.join(log_dir, "data_ingestion.log")
    file_handler = logging.FileHandler(log_file_path)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    console_handler.setFormatter(formatter)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# --------------------------------------------------
# 2. Load parameters from params.yaml
# --------------------------------------------------

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""

    try:
        with open(params_path, "r", encoding="utf-8") as file:
            params = yaml.safe_load(file)

        if not isinstance(params, dict):
            raise ValueError("The YAML file must contain a dictionary.")

        logger.debug("Parameters retrieved from %s", params_path)
        return params

    except Exception:
        logger.exception("Failed to load parameters from %s", params_path)
        raise


# --------------------------------------------------
# 3. Load dataset
# --------------------------------------------------

def load_data(data_url: str) -> pd.DataFrame:
    """Load data from a CSV file or URL."""

    try:
        df = pd.read_csv(data_url)

        logger.debug("Data loaded from %s", data_url)
        logger.info("Dataset shape: %s", df.shape)

        return df

    except Exception:
        logger.exception("Failed to load dataset from %s", data_url)
        raise


# --------------------------------------------------
# 4. Preprocess dataset
# --------------------------------------------------

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the spam dataset."""

    try:
        # Remove unnecessary columns only if they exist
        unwanted_columns = [
            "Unnamed: 2",
            "Unnamed: 3",
            "Unnamed: 4"
        ]

        existing_columns = [
            column for column in unwanted_columns
            if column in df.columns
        ]

        df = df.drop(columns=existing_columns)

        # Rename target and text columns
        rename_columns = {
            "v1": "target",
            "v2": "text"
        }

        df = df.rename(columns=rename_columns)

        # Validate the expected columns
        required_columns = ["target", "text"]
        missing_columns = [
            column for column in required_columns
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Dataset is missing required columns: {missing_columns}. "
                f"Available columns: {df.columns.tolist()}"
            )

        # Remove rows with missing target or text
        df = df.dropna(subset=required_columns)

        # Remove duplicate rows
        df = df.drop_duplicates()

        logger.debug("Data preprocessing completed")
        logger.info("Processed dataset shape: %s", df.shape)

        return df

    except Exception:
        logger.exception("Data preprocessing failed")
        raise


# --------------------------------------------------
# 5. Save training and testing datasets
# --------------------------------------------------

def save_data(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    data_path: str
) -> None:
    """Save training and testing datasets."""

    try:
        raw_data_path = os.path.join(data_path, "raw")
        os.makedirs(raw_data_path, exist_ok=True)

        train_file = os.path.join(raw_data_path, "train.csv")
        test_file = os.path.join(raw_data_path, "test.csv")

        train_data.to_csv(train_file, index=False)
        test_data.to_csv(test_file, index=False)

        logger.info("Training data saved to %s", train_file)
        logger.info("Testing data saved to %s", test_file)

    except Exception:
        logger.exception("Failed to save training or testing data")
        raise


# --------------------------------------------------
# 6. Main data ingestion pipeline
# --------------------------------------------------

def main():
    try:
        logger.info("Starting data ingestion process")

        # Load configuration
        params = load_params(params_path="params.yaml")

        # Read test_size from the YAML configuration
        test_size = params["data_ingestion"]["test_size"]

        if not isinstance(test_size, (int, float)):
            raise ValueError("test_size must be a number.")

        if not 0 < test_size < 1:
            raise ValueError("test_size must be between 0 and 1.")

        # Dataset URL
        data_url = (
            "https://raw.githubusercontent.com/"
            "vikashishere/Datasets/main/spam.csv"
        )

        # Load dataset
        df = load_data(data_url=data_url)

        # Preprocess dataset
        final_df = preprocess_data(df)

        if final_df.empty:
            raise ValueError("The dataset is empty after preprocessing.")

        # Split dataset into training and testing sets
        train_data, test_data = train_test_split(
            final_df,
            test_size=test_size,
            random_state=2,
            stratify=final_df["target"]
        )

        logger.info(
            "Data split completed: train=%d, test=%d",
            len(train_data),
            len(test_data)
        )

        # Save datasets
        save_data(
            train_data=train_data,
            test_data=test_data,
            data_path="./data"
        )

        logger.info("Data ingestion completed successfully")

    except Exception:
        logger.exception("Failed to complete the data ingestion process")
        raise


if __name__ == "__main__":
    main()
