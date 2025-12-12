from src.exception import Customexception
from src.logger import logging

from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact

import os
import sys
import numpy as np
import pandas as pd
import pymongo
from typing import List
from sklearn.model_selection import train_test_split
from dotenv import load_dotenv

load_dotenv()

MONGO_DB_URL = os.getenv("MONGO_DB_URL")


class DataIngestion:
    def __init__(self, data_ingestion_config: DataIngestionConfig):
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise Customexception(e, sys)

    # -------------------------------------------------------------------------------------
    # READ DATA FROM MONGODB
    # -------------------------------------------------------------------------------------
    def export_collection_as_dataframe(self) -> pd.DataFrame:
        """
        Connects to MongoDB, reads the collection, returns a DataFrame.
        Includes SSL handshake fixes for macOS & Python.
        """
        try:
            database_name = self.data_ingestion_config.database_name
            collection_name = self.data_ingestion_config.collection_name

            logging.info(f"Connecting to MongoDB cluster...")
            logging.info(f"DB: {database_name}, Collection: {collection_name}")

            # FIX FOR SSL HANDSHAKE ERROR
            self.mongo_client = pymongo.MongoClient(
                MONGO_DB_URL,
                tls=True,
                tlsAllowInvalidCertificates=True,
                serverSelectionTimeoutMS=30000,
                connectTimeoutMS=30000,
            )

            collection = self.mongo_client[database_name][collection_name]

            dataframe = pd.DataFrame(list(collection.find()))

            if dataframe.empty:
                raise Customexception(
                    f"MongoDB collection '{collection_name}' is EMPTY. No data to ingest.",
                    sys,
                )

            # Drop default MongoDB ID column
            if "_id" in dataframe.columns:
                dataframe.drop(columns=["_id"], inplace=True)

            # Replace string "na" with actual NaN
            dataframe.replace({"na": np.nan}, inplace=True)

            logging.info(f"DataFrame loaded successfully with shape: {dataframe.shape}")

            return dataframe

        except Exception as e:
            raise Customexception(e, sys)

    # -------------------------------------------------------------------------------------
    # SAVE RAW DATA TO FEATURE STORE
    # -------------------------------------------------------------------------------------
    def export_data_into_feature_store(self, dataframe: pd.DataFrame):
        try:
            feature_store_path = self.data_ingestion_config.feature_store_file_path
            dir_path = os.path.dirname(feature_store_path)

            os.makedirs(dir_path, exist_ok=True)

            dataframe.to_csv(feature_store_path, index=False)
            logging.info(f"Raw data saved to feature store: {feature_store_path}")

            return dataframe

        except Exception as e:
            raise Customexception(e, sys)

    # -------------------------------------------------------------------------------------
    # SPLIT TRAIN & TEST DATA
    # -------------------------------------------------------------------------------------
    def split_data_as_train_test(self, dataframe: pd.DataFrame):
        try:
            logging.info("Performing train-test split...")

            test_ratio = self.data_ingestion_config.train_test_split_ratio

            if len(dataframe) < 2:
                raise Customexception("Dataset too small to split.", sys)

            train_df, test_df = train_test_split(
                dataframe,
                test_size=test_ratio,
                shuffle=True,
                random_state=42
            )

            train_path = self.data_ingestion_config.training_file_path
            test_path = self.data_ingestion_config.testing_file_path

            os.makedirs(os.path.dirname(train_path), exist_ok=True)

            train_df.to_csv(train_path, index=False)
            test_df.to_csv(test_path, index=False)

            logging.info(f"Train/Test split completed.")
            logging.info(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

        except Exception as e:
            raise Customexception(e, sys)

    # -------------------------------------------------------------------------------------
    # ORCHESTRATE FULL INGESTION PIPELINE
    # -------------------------------------------------------------------------------------
    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("========== STARTING DATA INGESTION ==========")

            dataframe = self.export_collection_as_dataframe()
            dataframe = self.export_data_into_feature_store(dataframe)
            self.split_data_as_train_test(dataframe)

            artifact = DataIngestionArtifact(
                trained_file_path=self.data_ingestion_config.training_file_path,
                test_file_path=self.data_ingestion_config.testing_file_path,
            )

            logging.info("========== DATA INGESTION COMPLETED SUCCESSFULLY ==========")
            return artifact

        except Exception as e:
            raise Customexception(e, sys)
