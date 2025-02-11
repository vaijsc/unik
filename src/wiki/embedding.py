import logging
import os
from FlagEmbedding import FlagModel
from typing import Any
from src.utils import read_pickle, write_pickle

logger = logging.getLogger(__name__)

class EmbeddingModel:
    def __init__(
        self, 
        model_name_or_path: str, 
        normalize_embeddings: bool = True,
        use_fp16=True,
        trust_remote_code: bool = True, 
        ) -> None:
        """
        Initializes the embedding model.

        Parameters:
        - model_name_or_path (str): Path to the pre-trained model or model name.
        - trust_remote_code (bool): Whether to trust remote code for model initialization.
        - normalize_embeddings (bool): Whether to normalize the embeddings.
        """
        self.model = FlagModel(
            model_name_or_path, 
            use_fp16=use_fp16,
            normalize_embeddings=normalize_embeddings
        )
    
    def encode(
        self,
        texts: list[str],
        batch_size: int = 16,
        max_length: int = 1024,
        save_path: None | str = None
        ) -> list[Any]:
        if save_path and os.path.exists(save_path):
            logger.info(f"Load saved embeddings from `{save_path}`")
            return read_pickle(save_path, verbose=False)["embeddings"]
        
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            max_length=max_length,
        )
        if save_path:
            write_pickle(
                {"texts": texts, "embeddings": embeddings}, save_path, verbose=False
            )
            logger.info(f"Saved embedding to `{save_path}`")
        return embeddings
