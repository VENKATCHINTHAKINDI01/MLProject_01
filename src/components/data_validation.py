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
    Handles data validation steps:
    - Reading train/test files
    - Validating number of columns
    - Checking dataset drift
    - Saving validated data
    """

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact,
                 data_validation_config: DataValidationConfig):
        """
        Loads schema and initializes file paths.
        """
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config
            self._schema_config = read_yaml_file(SCHEMA_FILE_PATH)

            logging.info("Schema loaded successfully for validation.")

        except Exception as e:
            raise Customexception(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads a CSV file and returns a DataFrame.
        """
        try:
            logging.info(f"Reading file: {file_path}")
            return pd.read_csv(file_path)

        except Exception as e:
            raise Customexception(e, sys)

    def validate_number_of_columns(self, dataframe: pd.DataFrame) -> bool:
        """
        Validates whether the DataFrame contains expected number of columns.
        """
        try:
            required_columns = len(self._schema_config["columns"])
            existing_columns = len(dataframe.columns)

            logging.info(f"Expected columns: {required_columns}, Found: {existing_columns}")

            return required_columns == existing_columns

        except Exception as e:
            raise Customexception(e, sys)

    def detect_dataset_drift(self, base_df: pd.DataFrame, current_df: pd.DataFrame, threshold: float = 0.05) -> bool:
        """
        Detects dataset drift using Kolmogorov-Smirnov test.
        Saves drift report as YAML.
        """
        try:
            status = True
            drift_report = {}

            for column in base_df.columns:

                d1 = base_df[column]
                d2 = current_df[column]

                test_result = ks_2samp(d1, d2)

                # Drift occurs if p-value < threshold
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
            write_yaml_file(file_path=drift_report_file_path, content=drift_report)

            logging.info(f"Dataset drift report saved at: {drift_report_file_path}")
            return status

        except Exception as e:
            raise Customexception(e, sys)

    def initiate_data_validation(self) -> DataValidationArtifact:
        """
        Runs full data validation:
        - Reads train/test files
        - Validates structure
        - Detects drift
        - Saves validated datasets
        """
        try:
            logging.info("Starting Data Validation process...")

            # Read ingested data
            train_df = self.read_data(self.data_ingestion_artifact.trained_file_path)
            test_df = self.read_data(self.data_ingestion_artifact.test_file_path)

            # Validate columns in train
            if not self.validate_number_of_columns(train_df):
                raise Customexception("Train data column mismatch with schema.", sys)

            # Validate columns in test
            if not self.validate_number_of_columns(test_df):
                raise Customexception("Test data column mismatch with schema.", sys)

            # Detect dataset drift
            validation_status = self.detect_dataset_drift(train_df, test_df)

            # Save validated data
            os.makedirs(os.path.dirname(self.data_validation_config.valid_train_file_path), exist_ok=True)

            train_df.to_csv(self.data_validation_config.valid_train_file_path, index=False)
            test_df.to_csv(self.data_validation_config.valid_test_file_path, index=False)

            logging.info("Validated train/test files saved successfully.")

            # Build and return artifact
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
