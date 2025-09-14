import os
import pandas as pd
from google.cloud import storage
from src.logger import get_logger
from src.exception import CustomException
from config.path_config import *
from utils.common_function import read_yaml, load_data


logger = get_logger(__name__)

class DataIngestion:
    def __init__(self, config):
        self.config = config["data_ingetion"]
        self.bucket_name = self.config["bucket_name"]
        self.file_names = self.config["buckett_file_names"]
        os.makedirs(RAW_DIR, exist_ok=True)
        logger.info(f"Created directory: {RAW_DIR}")
        logger.info(f"DataIngestion started with config: {self.config}")
    
    def download_csv_from_gcp(self):
        try:
            client = storage.Client()
            bucket = client.bucket(self.bucket_name)
            for file_name in self.file_names:
                file_path = os.path.join(RAW_DIR, file_name)
                if file_name == "animelist.csv":
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    logger.info(f"Downloaded {file_name} to {file_path}")
                    data = pd.read_csv(file_path, nrows=5000000)
                    data.to_csv(file_path, index=False)
                    logger.info(f"Saved first 5,000,000 rows of {file_name} to {file_path}")
                else:
                    blob = bucket.blob(file_name)
                    blob.download_to_filename(file_path)
                    logger.info(f"Downloaded {file_name} to {file_path}")
        except Exception as e:
            logger.error(f"Error downloading files from GCP: {e}")
            raise CustomException("failed to download the data from gcp", e)
    
    def run(self):
        try:
            logger.info("Starting data ingestion process...")
            self.download_csv_from_gcp()
            logger.info("Data ingestion process completed successfully.")
        except Exception as e:
            logger.error(f"Error in data ingestion process: {e}")
            raise CustomException("failed to run the data ingestion process", e)
        finally:
            logger.info("Data ingestion process finished.")

if __name__ == "__main__":
    try:
        config = read_yaml(CONFIG_PATH)
        data_ingestion = DataIngestion(config)
        data_ingestion.run()
        logger.info("Data ingestion executed successfully.")
    except Exception as e:
        logger.error(f"Error in main execution: {e}")
        raise CustomException("failed in main execution", e)



