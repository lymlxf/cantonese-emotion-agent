"""
Text preprocessing utilities for Cantonese SER.
Converts Chinese text to word-level Word2Vec embeddings or CantoBERT token IDs.
"""

import torch
import numpy as np
from text2vec import Word2Vec
import jieba


class Word2VecTextPreprocessor:
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


class CantoBertTextPreprocessor:
    """
    Preprocess Chinese/Cantonese text to BART token IDs.

    Uses Ayaka/bart-base-cantonese tokenizer for Cantonese-specific
    subword tokenization. Only tokenizes — does NOT run the BART model
    (that happens in CantoBertEncoder).

    Args:
        model_name: HuggingFace model ID (default: "Ayaka/bart-base-cantonese")
        max_length: Maximum token sequence length (default: 128)
    """

    def __init__(self, model_name: str = "Ayaka/bart-base-cantonese", max_length: int = 128):
        print(f"Loading CantoBERT tokenizer: {model_name}...")
        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.max_length = max_length
        print("CantoBERT tokenizer loaded successfully!")

    def __call__(self, text: str) -> tuple:
        """
        Convert text to BART token IDs.

        Args:
            text: Chinese/Cantonese text string

        Returns:
            token_ids: numpy array of shape (seq_len,) dtype int64
            length: int, number of tokens (including special tokens)
        """
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # No padding here — collate_fn handles it
            return_tensors=None,  # Return python lists, not tensors
        )
        token_ids = np.array(encoding['input_ids'], dtype=np.int64)
        length = len(token_ids)
        return token_ids, length


class TextPreprocessor:
    """
    Backward-compatible wrapper: delegates to Word2Vec or CantoBERT preprocessor.

    Calling TextPreprocessor() without arguments defaults to Word2Vec mode,
    preserving backward compatibility with existing code.

    Args:
        text_encoder_type: "word2vec" (default) or "cantobert"
        **kwargs: Forwarded to the underlying preprocessor constructor
    """

    def __init__(self, text_encoder_type: str = "word2vec", **kwargs):
        self.text_encoder_type = text_encoder_type
        if text_encoder_type == "cantobert":
            cantobert_kwargs = {k: v for k, v in kwargs.items()
                                if k in ('model_name', 'max_length')}
            self._preprocessor = CantoBertTextPreprocessor(**cantobert_kwargs)
        else:
            word2vec_kwargs = {k: v for k, v in kwargs.items()
                               if k in ('model_name',)}
            self._preprocessor = Word2VecTextPreprocessor(**word2vec_kwargs)

    def __call__(self, text: str) -> tuple:
        return self._preprocessor(text)


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
