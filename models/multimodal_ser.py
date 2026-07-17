import torch
import torch.nn as nn

# Handle relative imports when running as script
if __name__ == "__main__" and __package__ is None:
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from models.audio_encoder import AudioEncoder
    from models.text_encoder import TextEncoder
    from models.cantobert_encoder import CantoBertEncoder
    from models.cross_modal_attention import CrossModalAttentionModule
    from models.statistics_pooling import StatisticsPooling
    from models.classifier import EmotionClassifier
else:
    from .audio_encoder import AudioEncoder
    from .text_encoder import TextEncoder
    from .cantobert_encoder import CantoBertEncoder
    from .cross_modal_attention import CrossModalAttentionModule
    from .statistics_pooling import StatisticsPooling
    from .classifier import EmotionClassifier


class MultimodalSER(nn.Module):
    """
    Multimodal Speech Emotion Recognition Model.
    
    Supports three modes:
    - "both": Full multimodal (audio + text with cross-modal attention)
    - "audio": Audio-only (no text encoder, no CMA)
    - "text": Text-only (no audio encoder, no CMA)
    
    When modality is "audio" or "text", the model can still load weights
    from a multimodal checkpoint using strict=False.
    """
    
    def __init__(
        self,
        audio_input_dim: int = 80,
        audio_hidden_dim: int = 128,
        text_hidden_dim: int = 128,
        d_model: int = 256,
        num_heads: int = 8,
        num_attention_blocks: int = 2,
        d_ff: int = 1024,
        classifier_hidden_dims = None,
        num_classes: int = 8,
        dropout: float = 0.2,
        freeze_embeddings: bool = True,
        modality: str = "both",
        text_encoder_type: str = "word2vec",
    ):
        """
        Args:
            modality: "both" | "audio" | "text"
                - "both": Full multimodal model
                - "audio": Audio-only model (smaller, faster inference)
                - "text": Text-only model (no audio processing needed)
            text_encoder_type: "word2vec" (default) | "cantobert"
                - "word2vec": Static Word2Vec + Conv1d + BiLSTM (existing)
                - "cantobert": Frozen BART-base-cantonese + Linear projection
        """
        super(MultimodalSER, self).__init__()
        
        assert modality in ["both", "audio", "text"], \
            f"modality must be 'both', 'audio', or 'text', got '{modality}'"
        
        self.modality = modality
        self.d_model = d_model
        self.num_classes = num_classes
        self.text_encoder_type = text_encoder_type
        
        # Create encoders based on modality
        if modality in ["both", "audio"]:
            self.audio_encoder = AudioEncoder(
                input_dim=audio_input_dim,
                hidden_dim=audio_hidden_dim,
                num_conv_layers=3,
                conv_channels=[64, 128, 256],
                kernel_size=3,
                dropout=dropout
            )
        
        if modality in ["both", "text"]:
            if text_encoder_type == "cantobert":
                self.text_encoder = CantoBertEncoder(
                    d_model=d_model,
                    freeze_backbone=True,
                )
            else:
                self.text_encoder = TextEncoder(
                    embedding_model_name="w2v-light-tencent-chinese",
                    hidden_dim=text_hidden_dim,
                    conv_channels=256,
                    kernel_size=3,
                    dropout=dropout,
                    freeze_embeddings=freeze_embeddings
                )
        
        # Cross-modal attention only for multimodal
        if modality == "both":
            # CMA-1: Audio as query, Text as key/value
            self.cma1 = CrossModalAttentionModule(
                d_model=d_model,
                num_heads=num_heads,
                num_blocks=num_attention_blocks,
                d_ff=d_ff,
                dropout=dropout
            )
            
            # CMA-2: Text as query, Audio as key/value
            self.cma2 = CrossModalAttentionModule(
                d_model=d_model,
                num_heads=num_heads,
                num_blocks=num_attention_blocks,
                d_ff=d_ff,
                dropout=dropout
            )
        
        self.statistics_pooling = StatisticsPooling()
        
        # Calculate classifier input dimension based on modality
        if modality == "both":
            # Concatenate two branches: d_model * 2 * 2 = d_model * 4
            classifier_input_dim = d_model * 4
        else:
            # Single branch: d_model * 2 (mean + std)
            classifier_input_dim = d_model * 2
        
        self.classifier = EmotionClassifier(
            input_dim=classifier_input_dim,
            hidden_dims=classifier_hidden_dims,
            num_classes=num_classes,
            dropout=dropout
        )

        # Audio-only classification head for auxiliary loss / stage-1 distillation.
        self.audio_head = nn.Linear(d_model * 2, num_classes)

        # Feature distillation projection: maps audio pooled features (d_model*2)
        # to text CLS token dimension (d_model). Used only in stage 1.
        self.distill_proj = nn.Linear(d_model * 2, d_model)
        
    def forward(
        self,
        audio_input: torch.Tensor = None,
        text_input: torch.Tensor = None,
        audio_lengths: torch.Tensor = None,
        text_lengths: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Forward pass. Input requirements depend on modality:
        - "both": audio_input and text_input required
        - "audio": audio_input required, text_input ignored
        - "text": text_input required, audio_input ignored
        """
        
        if self.modality in ["both", "audio"]:
            assert audio_input is not None, "audio_input is required for modality='{}'".format(self.modality)
            audio_features = self.audio_encoder(audio_input, audio_lengths)
        
        if self.modality in ["both", "text"]:
            assert text_input is not None, "text_input is required for modality='{}'".format(self.modality)
            if self.text_encoder_type == "cantobert":
                # text_input: (batch, seq_len) long — token IDs
                # Build attention mask from text_lengths (non-padded positions)
                if text_lengths is not None:
                    max_text_len = text_input.size(1)
                    attention_mask = torch.arange(max_text_len, device=text_input.device).unsqueeze(0) < text_lengths.unsqueeze(1)
                    attention_mask = attention_mask.long()
                else:
                    attention_mask = None
                text_features = self.text_encoder(text_input, attention_mask)
            else:
                text_features = self.text_encoder(text_input, text_lengths)
        
        if self.modality == "both":
            # Full multimodal path with cross-modal attention
            # Create attention masks
            audio_mask = None
            text_mask = None
            if audio_lengths is not None:
                batch_size, max_audio_len, _ = audio_features.shape
                audio_mask = torch.arange(max_audio_len, device=audio_features.device).unsqueeze(0) < audio_lengths.unsqueeze(1)
            if text_lengths is not None:
                batch_size, max_text_len, _ = text_features.shape
                text_mask = torch.arange(max_text_len, device=text_features.device).unsqueeze(0) < text_lengths.unsqueeze(1)
            
            # CMA-1: Audio query attends to Text key/value
            cma1_output = self.cma1(audio_features, text_features, text_features,
                                    query_mask=audio_mask, key_mask=text_mask)
            
            # CMA-2: Text query attends to Audio key/value
            cma2_output = self.cma2(text_features, audio_features, audio_features,
                                    query_mask=text_mask, key_mask=audio_mask)
            
            # Statistics Pooling
            audio_pooled = self.statistics_pooling(cma1_output, audio_lengths)
            text_pooled = self.statistics_pooling(cma2_output, text_lengths)
            
            # Concatenate
            combined = torch.cat([audio_pooled, text_pooled], dim=-1)
            
        elif self.modality == "audio":
            # Audio-only path
            audio_pooled = self.statistics_pooling(audio_features, audio_lengths)
            combined = audio_pooled
            
        elif self.modality == "text":
            # Text-only path
            text_pooled = self.statistics_pooling(text_features, text_lengths)
            combined = text_pooled
        
        # Classification
        logits = self.classifier(combined)
        
        return logits
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    import sys
    import os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    batch_size = 4
    n_mels = 80
    max_time_steps = 100
    max_text_len = 50
    
    # Create dummy inputs
    dummy_audio = torch.randn(batch_size, n_mels, max_time_steps)
    dummy_text = torch.randn(batch_size, max_text_len, 200)
    audio_lengths = torch.tensor([100, 60, 80, 40])
    text_lengths = torch.tensor([50, 30, 40, 20])
    
    print("=" * 60)
    print("Test 1: Multimodal (both)")
    print("=" * 60)
    model_both = MultimodalSER(
        audio_input_dim=n_mels,
        audio_hidden_dim=128,
        text_hidden_dim=128,
        d_model=256,
        num_heads=8,
        num_attention_blocks=2,
        num_classes=8,
        dropout=0.2,
        modality="both"
    )
    output_both = model_both(dummy_audio, dummy_text, audio_lengths, text_lengths)
    print(f"Output shape: {output_both.shape}")
    print(f"Parameters: {model_both.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("Test 2: Audio-only")
    print("=" * 60)
    model_audio = MultimodalSER(
        audio_input_dim=n_mels,
        audio_hidden_dim=128,
        d_model=256,
        num_classes=8,
        dropout=0.2,
        modality="audio"
    )
    output_audio = model_audio(dummy_audio, audio_lengths=audio_lengths)
    print(f"Output shape: {output_audio.shape}")
    print(f"Parameters: {model_audio.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("Test 3: Text-only")
    print("=" * 60)
    model_text = MultimodalSER(
        text_hidden_dim=128,
        d_model=256,
        num_classes=8,
        dropout=0.2,
        modality="text"
    )
    output_text = model_text(text_input=dummy_text, text_lengths=text_lengths)
    print(f"Output shape: {output_text.shape}")
    print(f"Parameters: {model_text.count_parameters():,}")
    
    print("\n" + "=" * 60)
    print("Test 4: Load multimodal weights into audio-only model")
    print("=" * 60)
    # Simulate: train multimodal, then deploy audio-only
    multimodal_state = model_both.state_dict()
    model_audio_deploy = MultimodalSER(
        audio_input_dim=n_mels,
        audio_hidden_dim=128,
        d_model=256,
        num_classes=8,
        dropout=0.2,
        modality="audio"
    )
    # Filter state dict to only include keys that exist in audio-only model
    # (classifier dimensions differ, so we skip mismatched keys)
    audio_state = {k: v for k, v in multimodal_state.items() 
                   if k in model_audio_deploy.state_dict() 
                   and v.shape == model_audio_deploy.state_dict()[k].shape}
    model_audio_deploy.load_state_dict(audio_state, strict=False)
    print(f"Loaded {len(audio_state)} matching parameter tensors")
    print("Audio encoder weights loaded successfully!")
    print("Note: Classifier needs separate training for unimodal deployment")
    
    # Verify it works
    output_deploy = model_audio_deploy(dummy_audio, audio_lengths=audio_lengths)
    print(f"Deployed model output shape: {output_deploy.shape}")
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Multimodal params:  {model_both.count_parameters():,}")
    print(f"Audio-only params:  {model_audio.count_parameters():,}")
    print(f"Text-only params:   {model_text.count_parameters():,}")
    print(f"\nAudio-only saves:   {model_both.count_parameters() - model_audio.count_parameters():,} parameters")
    print(f"Text-only saves:    {model_both.count_parameters() - model_text.count_parameters():,} parameters")
