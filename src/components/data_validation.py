from src.entity.artifact_entity import DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig
from src.exception import Customexception
from src.logger import logging 
from src.constants.training_pipeline import SCHEMA_FILE_PATH
from scipy.stats import ks_2samp
import pandas as pd
import os, sys
from src.utils.main_utils.utils import read_yaml_file, write_yaml_file


class DataValidation:
    """
    Handles full validation:
    - Schema validation
    - Numerical column validation
    - Drift detection
    - Saving validated files
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):

        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            logging.info("Schema loaded successfully.")

        except Exception as e:
            raise Customexception(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV and returns dataframe.
        """
        try:
            logging.info(f"Reading dataset from: {file_path}")
            return pd.read_csv(file_path)

        except Exception as e:
            raise Customexception(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates if the dataset contains the same number of columns as schema.
        """
        try:
            required_columns = len(self._schema_config["columns"])
            existing_columns = len(dataframe.columns)

            logging.info(f"Expected Columns: {required_columns}")
            logging.info(f"Existing Columns: {existing_columns}")

            return required_columns == existing_columns

        except Exception as e:
            raise Customexception(e, sys)


    def validate_numerical_columns_exist(self, dataframe: pd.DataFrame) -> bool:
        """
        Ensures all numerical columns defined in schema.yaml exist in dataframe.
        """
        try:
            required_numeric_cols = self._schema_config.get("numerical_columns", [])

            logging.info(f"Required numerical columns: {required_numeric_cols}")
            logging.info(f"Dataframe columns: {list(dataframe.columns)}")

            missing_columns = [
                col for col in required_numeric_cols if col not in dataframe.columns
            ]

            if missing_columns:
                logging.error(f"Missing numerical columns: {missing_columns}")
                return False

            logging.info("All required numerical columns are present.")
            return True

        except Exception as e:
            raise Customexception(e, sys)


    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame,
                             threshold: float = 0.05) -> bool:
        """
        Detects drift using KS test. Returns True if no significant drift detected.
        """
        try:
            status = True
            drift_report = {}

            for column in base_df.columns:

                d1 = base_df[column]
                d2 = current_df[column]

                test_result = ks_2samp(d1, d2)

                drift_detected = test_result.pvalue < threshold

                drift_report[column] = {
                    "p_value": float(test_result.pvalue),
                    "drift_detected": drift_detected
                }

                if drift_detected:
                    status = False

            # Save drift report
            drift_report_file_path = self.data_validation_config.drift_report_file_path
            os.makedirs(os.path.dirname(drift_report_file_path), exist_ok=True)
            write_yaml_file(drift_report_file_path, drift_report)

            logging.info(f"Drift report saved at: {drift_report_file_path}")
            return status

        except Exception as e:
            raise Customexception(e, sys)


    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Full data validation pipeline.
        """
        try:
            logging.info("Starting Data Validation pipeline.")

            # Load ingested data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # 1️⃣ Validate column count
            if not self.validate_number_of_columns(train_df):
                raise Customexception("Train file column mismatch.", sys)

            if not self.validate_number_of_columns(test_df):
                raise Customexception("Test file column mismatch.", sys)

            # 2️⃣ Validate numerical column presence
            if not self.validate_numerical_columns_exist(train_df):
                raise Customexception("Train file missing numerical columns.", sys)

            if not self.validate_numerical_columns_exist(test_df):
                raise Customexception("Test file missing numerical columns.", sys)

            # 3️⃣ Drift Detection
            validation_status = self.detect_dataset_drift(train_df, test_df)

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            logging.info("Validated train & test files saved successfully.")

            # Build artifact
            data_validation_artifact = DataValidationArtifact(
                validation_status=validation_status,
                valid_train_file_path=self.data_validation_config.valid_train_file_path,
                valid_test_file_path=self.data_validation_config.valid_test_file_path,
                invalid_train_file_path=None,
                invalid_test_file_path=None,
                drift_report_file_path=self.data_validation_config.drift_report_file_path,
            )

            logging.info("Data Validation completed successfully.")
            return data_validation_artifact

        except Exception as e:
            raise Customexception(e, sys)
