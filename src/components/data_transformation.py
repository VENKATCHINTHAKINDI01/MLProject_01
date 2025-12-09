import sys
import os
import numpy as np
import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline

from src.constants.training_pipeline import TARGET_COLUMN
from src.constants.training_pipeline import DATA_TRANSFORMATION_IMPUTER_PARAMS

from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataValidationArtifact
)

from src.entity.config_entity import DataTransformationConfig
from src.exception import Customexception
from src.logger import logging
from src.utils.main_utils.utils import (
    save_numpy_array_data,
    save_object
)


class DataTransformation:
    """
    Performs:
    - KNN Imputation
    - Train/Test transformation
    - Saves numpy arrays & preprocessing object
    """

    def __init__(self, data_validation_artifact: DataValidationArtifact,
                 data_transformation_config: DataTransformationConfig):

        try:
            self.data_validation_artifact = data_validation_artifact
            self.data_transformation_config = data_transformation_config

        except Exception as e:
            raise Customexception(e, sys)

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        """
        Reads CSV file and returns DataFrame.
        """
        try:
            logging.info(f"Reading dataset from: {file_path}")
            return pd.read_csv(file_path)

        except Exception as e:
            raise Customexception(e, sys)

    # ----------------------------------------------------------------------------------------
    # Create Transformation Pipeline
    # ----------------------------------------------------------------------------------------
    @classmethod
    def get_data_transformer_object(cls) -> Pipeline:
        """
        Creates KNN Imputer pipeline.
        """
        try:
            logging.info("Initializing KNNImputer pipeline...")

            imputer = KNNImputer(**DATA_TRANSFORMATION_IMPUTER_PARAMS)

            processor = Pipeline([
                ("imputer", imputer)
            ])

            logging.info("KNNImputer pipeline created.")
            return processor

        except Exception as e:
            raise Customexception(e, sys)

    # ----------------------------------------------------------------------------------------
    # Main Transformation Logic
    # ----------------------------------------------------------------------------------------
    def initiate_data_transformation(self) -> DataTransformationArtifact:
        logging.info("===== Starting Data Transformation Stage =====")

        try:
            # ---------------------------------------------------------
            # Step 1: Read validated train/test files
            # ---------------------------------------------------------
            train_df = self.read_data(self.data_validation_artifact.valid_train_file_path)
            test_df = self.read_data(self.data_validation_artifact.valid_test_file_path)

            if TARGET_COLUMN not in train_df.columns:
                raise Customexception(f"Target column '{TARGET_COLUMN}' not found in training dataset", sys)

            if TARGET_COLUMN not in test_df.columns:
                raise Customexception(f"Target column '{TARGET_COLUMN}' not found in test dataset", sys)

            # ---------------------------------------------------------
            # Step 2: Split input & output features
            # ---------------------------------------------------------
            input_feature_train_df = train_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_train_df = train_df[TARGET_COLUMN].replace(-1, 0)

            input_feature_test_df = test_df.drop(columns=[TARGET_COLUMN], axis=1)
            target_feature_test_df = test_df[TARGET_COLUMN].replace(-1, 0)

            # ---------------------------------------------------------
            # Step 3: Get preprocessing pipeline
            # ---------------------------------------------------------
            preprocessor = self.get_data_transformer_object()

            preprocessor_object = preprocessor.fit(input_feature_train_df)

            transformed_input_train_feature = preprocessor_object.transform(input_feature_train_df)
            transformed_input_test_feature = preprocessor_object.transform(input_feature_test_df)

            # Combine X + y
            train_arr = np.c_[transformed_input_train_feature, np.array(target_feature_train_df)]
            test_arr = np.c_[transformed_input_test_feature, np.array(target_feature_test_df)]

            # ---------------------------------------------------------
            # Step 4: Save arrays & preprocessor object
            # ---------------------------------------------------------
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_train_file_path), exist_ok=True)
            os.makedirs(os.path.dirname(self.data_transformation_config.transformed_object_file_path), exist_ok=True)

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_train_file_path,
                array=train_arr
            )

            save_numpy_array_data(
                file_path=self.data_transformation_config.transformed_test_file_path,
                array=test_arr
            )

            save_object(
                file_path=self.data_transformation_config.transformed_object_file_path,
                obj=preprocessor_object
            )

            # Also save preprocessor for final model use
            save_object("final_model/preprocessor.pkl", preprocessor_object)

            # ---------------------------------------------------------
            # Step 5: Create Artifact
            # ---------------------------------------------------------
            artifact = DataTransformationArtifact(
                transformed_object_file_path=self.data_transformation_config.transformed_object_file_path,
                transformed_train_file_path=self.data_transformation_config.transformed_train_file_path,
                transformed_test_file_path=self.data_transformation_config.transformed_test_file_path
            )

            logging.info("===== Data Transformation Completed Successfully =====")
            return artifact

        except Exception as e:
            raise Customexception(e, sys)
