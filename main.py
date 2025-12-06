from src.components.data_ingestion import DataIngestion
from src.exception import Customexception
from src.logger import logging
from src.entity.config_entity import DataIngestionConfig
from src.entity.config_entity import TrainingPipelineConfig

import sys

if __name__== "__main__":
    
    try:
        trainingpipelineconfig=TrainingPipelineConfig()
        dataingestionconfig = DataIngestionConfig(trainingpipelineconfig)
        data_ingestion = DataIngestion(dataingestionconfig)
        logging.info("initiate the data ingestion")
        dataingestionartifact=data_ingestion.initiate_data_ingestion()
        print(dataingestionartifact)
        
        
    except Exception as e:
        raise Customexception(e, sys)