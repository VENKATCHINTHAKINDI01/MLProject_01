import os
import sys
import json
import pandas as pd
import numpy as np
import pymongo
import certifi
from dotenv import load_dotenv

from src.exception import Customexception
from src.logger import logging

from urllib.parse import quote_plus
from pymongo import MongoClient

load_dotenv()

# USE ATLAS CONNECTION DIRECTLY
MONGO_DB_URL = "mongodb+srv://vyaschinthakindi:Venkat123@cluster0.cmqoumo.mongodb.net/?appName=Cluster0"

class NetworkDataExtract:
    def csv_to_json_convertor(self, file_path):
        try:
            data = pd.read_csv(file_path)
            data.reset_index(drop=True, inplace=True)
            records = list(json.loads(data.T.to_json()).values())
            return records
        except Exception as e:
            raise Customexception(e, sys)
        
    def insert_data_mongodb(self, records, database, collection):
        try:
            # FIXED CONNECTION
            mongo_client = pymongo.MongoClient(MONGO_DB_URL)

            db = mongo_client[database]
            col = db[collection]

            col.insert_many(records)

            return len(records)
        except Exception as e:
            raise Customexception(e, sys)
        
if __name__ == '__main__':
    FILE_PATH = os.path.join("NETWORK_data", "phisingData.csv")
    DATABASE = "vyasAI"
    COLLECTION = "NetworkData"

    networkobj = NetworkDataExtract()
    records = networkobj.csv_to_json_convertor(FILE_PATH)
    print(f"Records to insert: {len(records)}")

    no_of_records = networkobj.insert_data_mongodb(records, DATABASE, COLLECTION)
    
    print(f"Inserted {no_of_records} records successfully.")
