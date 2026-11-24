"""Shared compact SleepNetZero configuration for respiratory event detection."""

from models.base import base_model

ARCHITECTURE = {
    "embedding_size": 64,
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
}
AHI_TOKENS = {"heartbreath": 50, "snore": 250}


def build_event_model():
    features = [dict(head=name, token_len=size, input_channels=1) for name, size in AHI_TOKENS.items()]
    backbone = dict(ARCHITECTURE, features=features, _name="models.sleep_net_zero.SleepNetZeroBackbone")
    head = dict(
        _name="models.heads.seqProj", hidden_size=ARCHITECTURE["hidden_size"],
        output_name="ahi_event", num_classes=1,
    )
    return base_model(backbone, [head])
