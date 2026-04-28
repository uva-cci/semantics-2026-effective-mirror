import logging

import torch
import torch.nn.functional as F
from sentence_transformers import SentenceTransformer

from src.config import EmbeddingModelConfig


def score_vectors(
    a: torch.Tensor,
    b: torch.Tensor,
) -> float:
    """
    Compute a pairwise score between two embedding vectors.

    Parameters
    ----------
    a, b : torch.Tensor
        Embedding vectors.

    Returns
    -------
    float
        The computed score.
    """
    a_norm = F.normalize(a, p=2, dim=-1)
    b_norm = F.normalize(b, p=2, dim=-1)
    return float(torch.dot(a_norm, b_norm))  # pyright: ignore[reportPrivateImportUsage]


def get_encoder(cfg: EmbeddingModelConfig) -> SentenceTransformer:
    fullname = f"{cfg.org}/{cfg.name}"
    logging.info(
        f"Loading encoder {fullname} (downloading from HF Hub if not cached)")
    return SentenceTransformer(fullname, trust_remote_code=True)
