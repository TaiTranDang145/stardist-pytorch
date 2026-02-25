import torch
import torch.nn.functional as F

def generic_masked_loss(mask, loss_fn, weights=1.0, norm_by_mask=True, reg_weight=0.0, reg_penalty='abs'):
    """
    Wrapper loss masked chung (tương đương generic_masked_loss gốc).
    
    Args:
        mask: torch.Tensor (B, H, W, 1) hoặc (B, H, W) - mask valid (1=valid, 0=ignore)
        loss_fn: callable(y_true, y_pred) -> per-pixel loss
        weights: float hoặc tensor, trọng số cho từng pixel
        norm_by_mask: bool - có chuẩn hóa theo % valid pixels không
        reg_weight: float - hệ số phạt background
        reg_penalty: 'abs' hoặc 'square' - hàm phạt cho y_pred ở vùng mask=0
    
    Returns:
        loss callable: (y_true, y_pred) -> scalar loss
    """
    def _loss(y_true, y_pred):
        # Cast về float
        mask = mask.float()
        weights = torch.as_tensor(weights, device=y_true.device, dtype=torch.float32)

        # Actual loss chỉ trên vùng mask=1
        per_pixel_loss = loss_fn(y_true, y_pred)
        actual_loss = torch.mean(mask * weights * per_pixel_loss, dim=[1,2,3])  # mean theo spatial

        # Normalize theo % valid pixels
        norm_mask = (torch.mean(mask, dim=[1,2,3]) + 1e-8) if norm_by_mask else 1.0
        normalized_loss = actual_loss / norm_mask

        # Background regularization (nếu reg_weight > 0)
        if reg_weight > 0:
            if reg_penalty == 'abs':
                reg_fn = torch.abs
            elif reg_penalty == 'square':
                reg_fn = torch.square
            else:
                raise ValueError("reg_penalty chỉ hỗ trợ 'abs' hoặc 'square'")

            reg_loss = torch.mean((1 - mask) * reg_fn(y_pred), dim=[1,2,3])
            total_loss = normalized_loss + reg_weight * reg_loss
        else:
            total_loss = normalized_loss

        return total_loss.mean()  # mean theo batch

    return _loss


def masked_mae_loss(mask, reg_weight=1e-4, norm_by_mask=True):
    """
    Masked MAE cho dist head (tương đương masked_loss_mae gốc)
    """
    def mae_loss(y_true, y_pred):
        return torch.abs(y_true - y_pred)

    return generic_masked_loss(
        mask=mask,
        loss_fn=mae_loss,
        weights=1.0,
        norm_by_mask=norm_by_mask,
        reg_weight=reg_weight,
        reg_penalty='abs'   # gốc dùng K.abs
    )


def masked_bce_loss():
    """
    Masked BCE cho prob head (tương đương prob_loss gốc)
    """
    def bce(y_true, y_pred):
        # Mask ignore: y_true < 0
        valid_mask = (y_true >= 0).float()
        y_true = torch.clamp(y_true, 0.0, 1.0)
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)

        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        masked_loss = torch.sum(bce_loss * valid_mask) / (torch.sum(valid_mask) + 1e-8)
        return masked_loss

    return bce


def kld_metric(y_true, y_pred):
    """
    KL divergence metric cho prob head (tương đương kld gốc)
    Dùng để theo dõi như paper, không phải loss chính
    """
    valid_mask = (y_true >= 0)
    y_true_valid = torch.clamp(y_true[valid_mask], 1e-7, 1.0 - 1e-7)
    y_pred_valid = torch.clamp(y_pred[valid_mask], 1e-7, 1.0 - 1e-7)

    bce_pred = F.binary_cross_entropy(y_pred_valid, y_true_valid, reduction='mean')
    bce_true = F.binary_cross_entropy(y_true_valid, y_true_valid, reduction='mean')

    return bce_pred - bce_true


# ====================== Tổng loss (dùng trong training) ======================
def total_loss(prob_pred, dist_pred, prob_gt, dist_mask_gt, loss_weights=(1.0, 0.2)):
    """
    Tổng loss = w1 * prob_loss + w2 * dist_loss
    dist_mask_gt: (B, H, W, n_rays + 1) - kênh cuối là mask
    """
    prob_loss_fn = masked_bce_loss()
    dist_loss_fn = masked_mae_loss(
        mask=dist_mask_gt[..., -1:],  # kênh mask cuối
        reg_weight=1e-4,
        norm_by_mask=True
    )

    prob_loss = prob_loss_fn(prob_gt, prob_pred)
    dist_loss = dist_loss_fn(dist_mask_gt[..., :-1], dist_pred)  # dist true là n_rays kênh đầu

    total = loss_weights[0] * prob_loss + loss_weights[1] * dist_loss
    return total