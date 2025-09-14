from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Dot, BatchNormalization, Activation
from utils.common_function import read_yaml
from src.logger import get_logger
from src.exception import CustomException
import sys

logger = get_logger(__name__)

class BaseModel:
    def __init__(self, config_path):
        try:
            self.config = read_yaml(config_path)
        except Exception as e:
            raise CustomException(e, sys)
    def RecomenderNet(self, n_users, n_anime):
        try:
            embedding_size = self.config["model"]["embedding_size"]
            user = Input(name="user", shape = [1])
            user_embedding = Embedding(name="user_embedding", input_dim=n_users, output_dim=embedding_size)(user)
            anime = Input(name="anime", shape = [1])
            anime_embedding = Embedding(name="anime_embedding", input_dim=n_anime, output_dim=embedding_size)(anime)
            x = Dot(name="dot_product", normalize = True, axes = 2)([user_embedding, anime_embedding])
            x= Flatten()(x)
            x = Dense(1, kernel_initializer="he_normal")(x)
            x = BatchNormalization()(x)
            x = Activation("sigmoid")(x)
            model = Model(inputs=[user, anime], outputs=x)
            model.compile(
                loss = self.config["model"]["loss"],
                optimizer = self.config["model"]["optimizer"],
                metrics = self.config["model"]["metrics"]
            )
            logger.info("Model compiled successfully.")
            return model   
        except Exception as e:
            logger.error(f"Error compiling model: {e}")
            raise CustomException(e, sys)
        
