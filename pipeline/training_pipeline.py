from utils.common_function import read_yaml
from src.data_ingestion import DataIngestion
from src.data_processing import DataProcessor
from src.model_training import ModelTraining
from config.path_config import *


if __name__ == "__main__":
        data_processor = DataProcessor(ANIMELIST_CSV, PROCESSED_DIR)
        data_processor.run()
        model_training = ModelTraining(data_path=PROCESSED_DIR)
        model_training.train_model()
