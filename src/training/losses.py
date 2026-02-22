import torch
import torch.nn.functional as F


def clip_loss(
    logits_per_image: torch.Tensor,
    logits_per_text: torch.Tensor,
) -> torch.Tensor:
    """Symmetric cross-entropy contrastive loss (CLIP objective).

    Args:
        logits_per_image: (B, B) similarity matrix — rows are images.
        logits_per_text:  (B, B) similarity matrix — rows are texts.

    Returns:
        Scalar loss tensor.
    """
    batch_size = logits_per_image.shape[0]
    labels = torch.arange(batch_size, device=logits_per_image.device)

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)

    return (loss_i + loss_t) / 2
