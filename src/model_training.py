import joblib
import comet_ml
import numpy as np
import os
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, LearningRateScheduler
from src.base_model import BaseModel
from src.logger import get_logger
from src.exception import CustomException
import sys
from config.path_config import *    

logger = get_logger(__name__)

class ModelTraining:
    def __init__(self, data_path):
        self.data_path = data_path
        # Initialize Comet Experiment
        self.experiment = comet_ml.Experiment(
            api_key="ePI9nxiE2TqyhqDvRxV9VWdV1",
            project_name="mlops-course2",
            workspace="jayesh-naik"
        )
        logger.info("Model Training started...")

    def load_data(self):
        try:
            X_train_array = joblib.load(X_TRAIN_ARRAY)
            X_test_array = joblib.load(X_TEST_ARRAY)
            y_train = joblib.load(Y_TRAIN)
            y_test = joblib.load(Y_TEST)
            logger.info("Data loaded successfully.")
            return X_train_array, X_test_array, y_train, y_test
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise CustomException("Failed to load the data", sys)
    
    def train_model(self):
        try:
            X_train_array, X_test_array, y_train, y_test = self.load_data()
            n_users = len(joblib.load(USER2USER_ENCODED))
            n_anime = len(joblib.load(ANIME2ANIME_ENCODED))

            base_model = BaseModel(config_path=CONFIG_PATH)
            model = base_model.RecomenderNet(n_users, n_anime)

            # Hyperparameters
            params = {
                "start_lr": 1e-5,
                "min_lr": 1e-6,
                "max_lr": 5e-5,
                "batch_size": 10000,
                "epochs": 20,
                "rampup_epochs": 5,
                "sustained_epochs": 0,
                "exp_decay": 0.8,
            }

            # Log hyperparameters to Comet
            self.experiment.log_parameters(params)

            def lrfn(epoch):
                if epoch < params["rampup_epochs"]:
                    return (params["max_lr"] - params["start_lr"]) / params["rampup_epochs"] * epoch + params["start_lr"]
                elif epoch < params["rampup_epochs"] + params["sustained_epochs"]:
                    return params["max_lr"]
                else:
                    return (params["max_lr"] - params["min_lr"]) * params["exp_decay"]**(epoch - params["rampup_epochs"] - params["sustained_epochs"]) + params["min_lr"]
            
            lr_callback = LearningRateScheduler(lrfn, verbose=0)

            model_checkpoint = ModelCheckpoint(
                filepath=CHECK_POINT_FILE_PATH,
                save_weights_only=True,
                monitor="val_loss",
                mode="min",
                save_best_only=True
            )

            early_stopping = EarlyStopping(
                patience=3,
                monitor="val_loss",
                mode="min",
                restore_best_weights=True
            )

            my_callbacks = [lr_callback, early_stopping, model_checkpoint]

            os.makedirs(os.path.dirname(CHECK_POINT_FILE_PATH), exist_ok=True)
            os.makedirs(MODEL_DIR, exist_ok=True)
            os.makedirs(WEIGHTS_DIR, exist_ok=True)

            # Train model
            history = model.fit(
                x=X_train_array,
                y=y_train,
                batch_size=params["batch_size"],
                epochs=params["epochs"],
                verbose=1, 
                validation_data=(X_test_array, y_test),
                callbacks=my_callbacks
            )

            model.load_weights(CHECK_POINT_FILE_PATH)

            # Log training and validation losses per epoch
            for epoch in range(len(history.history['loss'])):
                self.experiment.log_metric("train_loss", history.history['loss'][epoch], step=epoch)
                self.experiment.log_metric("val_loss", history.history['val_loss'][epoch], step=epoch)

            logger.info("Model trained successfully.")
            self.save_model_weights(model)

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise CustomException("Failed to train the model", sys)
        
    def extract_weights(self, layer_name, model):
        try:
            weight_layer = model.get_layer(layer_name)
            weights = weight_layer.get_weights()[0]
            weights = weights / np.linalg.norm(weights, axis=1, keepdims=True)
            logger.info(f"Weights from {layer_name} extracted successfully.")
            return weights
        except Exception as e:
            logger.error(f"Error extracting weights: {e}")
            raise CustomException("Failed to extract the weights", sys)
        
    def save_model_weights(self, model):
        try:
            model.save(MODEL_PATH)
            logger.info("Model saved successfully.")

            user_weights = self.extract_weights("user_embedding", model)
            anime_weights = self.extract_weights("anime_embedding", model)

            joblib.dump(user_weights, USER_WEIGHTS_PATH)
            joblib.dump(anime_weights, ANIME_WEIGHTS_PATH)

            # Log assets to Comet
            self.experiment.log_asset(MODEL_PATH)
            self.experiment.log_asset(USER_WEIGHTS_PATH)
            self.experiment.log_asset(ANIME_WEIGHTS_PATH)

            logger.info("Weights saved successfully.")
        except Exception as e:
            logger.error(f"Error saving model weights: {e}")
            raise CustomException("Failed to save the model weights", sys)


if __name__ == "__main__":
    model_training = ModelTraining(data_path=PROCESSED_DIR)
    model_training.train_model()
    
