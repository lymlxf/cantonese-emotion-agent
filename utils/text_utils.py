"""
Text preprocessing utilities for Cantonese SER.
Converts Chinese text to word-level Word2Vec embeddings.
"""

import torch
import numpy as np
from text2vec import Word2Vec
import jieba


class TextPreprocessor:
    """
    Preprocess Chinese/Cantonese text to word-level Word2Vec embeddings.
    
    Uses jieba for word segmentation and text2vec for word embedding lookup.
    
    Args:
        model_name: Name of the text2vec model (default: "w2v-light-tencent-chinese")
    """
    
    def __init__(self, model_name: str = "w2v-light-tencent-chinese"):
        print(f"Loading Word2Vec model: {model_name}...")
        self.w2v_model = Word2Vec(model_name)
        self.embedding_dim = 200  # Tencent Word2Vec dimension
        print("Word2Vec model loaded successfully!")
    
    def __call__(self, text: str) -> tuple:
        """
        Convert text to word-level Word2Vec embeddings.
        
        Args:
            text: Chinese/Cantonese text string
            
        Returns:
            embeddings: numpy array of shape (seq_len, 200)
            length: int, number of words
        """
        # 1. Segment text into words using jieba
        words = list(jieba.cut(text))
        
        # 2. Encode each word to embedding
        # text2vec.encode() supports list of words
        embeddings = self.w2v_model.encode(words, show_progress_bar=False)
        
        # embeddings shape: (num_words, 200)
        length = embeddings.shape[0]
        
        return embeddings, length


if __name__ == "__main__":
    # Test with sample data
    import pandas as pd
    
    print("Testing TextPreprocessor (word-level)...")
    
    # Load sample parquet
    df = pd.read_parquet("data/train-00000-of-00045.parquet")
    
    # Initialize preprocessor
    preprocessor = TextPreprocessor()
    
    # Test with a few samples
    print("\nTesting word-level encoding...")
    for i in range(5):
        text = df['text'].iloc[i]
        embeddings, length = preprocessor(text)
        
        print(f"Sample {i}:")
        print(f"  Text length (chars): {len(text)}")
        print(f"  Word count: {length}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Value range: [{embeddings.min():.2f}, {embeddings.max():.2f}]")
        print()
