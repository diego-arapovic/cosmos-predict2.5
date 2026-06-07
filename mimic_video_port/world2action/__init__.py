"""world2action overlay for Cosmos-Predict2.5.

Ports mimic-video's world2action method (frozen Video2World backbone + a separate
cross-attention action decoder) onto the cosmos-predict2.5 stack. See ../README.md.
"""

from .checkpoints import (
    REASON1_UUID,
    WAN_VAE_PATH,
    register_external_checkpoints,
)

__all__ = ["register_external_checkpoints", "REASON1_UUID", "WAN_VAE_PATH"]
