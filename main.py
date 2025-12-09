from src.components.data_ingestion import DataIngestion
from src.components.data_validation import DataValidation
from src.exception import Customexception
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig,DataValidationConfig
from src.entity.config_entity import TrainingPipelineConfig

import sys

if __name__== "__main__":
    
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        logging.info("data initiation is completed")
        print(dataingestionartifact)
        data_validation_config = DataValidationConfig(trainingpipelineconfig)
        data_validation= DataValidation(dataingestionartifact,data_validation_config)
        logging.info("initiate the data validation process")
        data_validation_artifact= data_validation.initiate_data_validation()
        logging.info("the data validation completed ")
       
        
        
    except Exception as e:
        raise Customexception(e, sys)