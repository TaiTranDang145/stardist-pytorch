import torch
import torch.nn.functional as F

def generic_masked_loss(mask, loss_fn, weights=1.0, norm_by_mask=True, reg_weight=0.0, reg_penalty='abs'):
    """
    Wrapper loss masked chung (tương đương generic_masked_loss gốc).
    """
    def _loss(y_true, y_pred):
        m = mask.float()
        w = torch.as_tensor(weights, device=y_true.device, dtype=torch.float32)
 
        per_pixel_loss = loss_fn(y_true, y_pred)
        actual_loss = torch.mean(m * w * per_pixel_loss, dim=[1,2,3])
 
        norm_mask = (torch.mean(m, dim=[1,2,3]) + 1e-8) if norm_by_mask else 1.0
        normalized_loss = actual_loss / norm_mask
 
        if reg_weight > 0:
            if reg_penalty == 'abs':
                reg_fn = torch.abs
            elif reg_penalty == 'square':
                reg_fn = torch.square
            else:
                raise ValueError("reg_penalty chỉ hỗ trợ 'abs' hoặc 'square'")
 
            reg_loss = torch.mean((1 - m) * reg_fn(y_pred), dim=[1,2,3])
            total_loss_val = normalized_loss + reg_weight * reg_loss
        else:
            total_loss_val = normalized_loss
 
        return total_loss_val.mean()

    return _loss


def masked_mae_loss(mask, reg_weight=1e-4, norm_by_mask=True):
    def mae_loss(y_true, y_pred):
        return torch.abs(y_true - y_pred)

    return generic_masked_loss(
        mask=mask,
        loss_fn=mae_loss,
        weights=1.0,
        norm_by_mask=norm_by_mask,
        reg_weight=reg_weight,
        reg_penalty='abs'
    )


def masked_bce_loss():
    def bce(y_true, y_pred):
        valid_mask = (y_true >= 0).float()
        y_true = torch.clamp(y_true, 0.0, 1.0)
        y_pred = torch.clamp(y_pred, 1e-7, 1.0 - 1e-7)

        bce_loss = F.binary_cross_entropy(y_pred, y_true, reduction='none')
        masked_loss = torch.sum(bce_loss * valid_mask) / (torch.sum(valid_mask) + 1e-8)
        return masked_loss

    return bce


def boundary_dice_loss(y_true, y_pred, smooth=1e-5):
    """
    Boundary Dice Loss để xử lý class imbalance cho boundary head.
    
    Args:
        y_true: Ground truth boundary mask (B, 1, H, W) with values in {0, 1}
        y_pred: Predicted boundary probability (B, 1, H, W) with values in [0, 1]
        smooth: Laplace smoothing để tránh chia cho 0
    
    Returns:
        Dice Loss = 1 - Dice Coefficient
    
    Công thức:
        Dice = (2 * |X ∩ Y| + ε) / (|X| + |Y| + ε)
        Loss = 1 - Dice
    """
    # Flatten spatial dimensions
    y_true_f = y_true.view(y_true.size(0), -1)  # (B, H*W)
    y_pred_f = y_pred.view(y_pred.size(0), -1)  # (B, H*W)
    
    # Tính intersection và union
    intersection = torch.sum(y_pred_f * y_true_f, dim=1)  # (B,)
    
    # Tổng các pixel
    sum_pred = torch.sum(y_pred_f, dim=1)  # (B,)
    sum_true = torch.sum(y_true_f, dim=1)  # (B,)
    
    # Dice coefficient
    dice = (2.0 * intersection + smooth) / (sum_pred + sum_true + smooth)
    
    # Dice loss
    loss = 1.0 - dice
    
    return loss.mean()


def kld_metric(y_true, y_pred):
    valid_mask = (y_true >= 0)
    y_true_valid = torch.clamp(y_true[valid_mask], 1e-7, 1.0 - 1e-7)
    y_pred_valid = torch.clamp(y_pred[valid_mask], 1e-7, 1.0 - 1e-7)

    bce_pred = F.binary_cross_entropy(y_pred_valid, y_true_valid, reduction='mean')
    bce_true = F.binary_cross_entropy(y_true_valid, y_true_valid, reduction='mean')

    return bce_pred - bce_true


def masked_fourier_loss(mask):
    def fourier_mae(y_true, y_pred):
        return torch.abs(y_true - y_pred)

    return generic_masked_loss(
        mask=mask,
        loss_fn=fourier_mae,
        reg_weight=1e-4
    )


def complexity_loss_fn(mask):
    def bce_loss(y_true, y_pred):
        return F.binary_cross_entropy(y_pred, y_true, reduction='none')

    return generic_masked_loss(
        mask=mask,
        loss_fn=bce_loss
    )


# Total loss với Boundary Dice Loss (theo yêu cầu CHANGES.md)
def total_loss_with_boundary(prob_pred, dist_pred, boundary_pred, 
                             prob_gt, dist_mask_gt, boundary_gt,
                             loss_weights=(1.0, 0.2, 0.5)):
    """
    Combined Loss Function với 3 heads như yêu cầu trong CHANGES.md:
    Total Loss = λ1·L_BCE + λ2·L_MAE + λ3·L_Dice_Boundary
    
    Args:
        prob_pred: Probability prediction (B, 1, H, W)
        dist_pred: Distance prediction (B, n_rays, H, W)
        boundary_pred: Boundary prediction (B, 1, H, W)
        prob_gt: Ground truth probability (B, 1, H, W)
        dist_mask_gt: Distance + mask (B, n_rays+1, H, W) - kênh cuối là mask
        boundary_gt: Ground truth boundary (B, 1, H, W)
        loss_weights: (λ1, λ2, λ3) - trọng số cho từng loss
    
    Returns:
        total_loss, (p_loss, d_loss, b_loss)
    """
    # 1. Probability Loss (BCE)
    prob_loss_fn = masked_bce_loss()
    p_loss = prob_loss_fn(prob_gt, prob_pred)
    
    # 2. Distance Loss (MAE)
    mask = dist_mask_gt[:, -1:]  # Lấy mask từ kênh cuối
    dist_gt = dist_mask_gt[:, :-1]  # Lấy distance rays
    dist_loss_fn = masked_mae_loss(mask=mask)
    d_loss = dist_loss_fn(dist_gt, dist_pred)
    
    # 3. Boundary Dice Loss (NEW)
    b_loss = boundary_dice_loss(boundary_gt, boundary_pred)
    
    # Combined loss
    total = (loss_weights[0] * p_loss + 
             loss_weights[1] * d_loss + 
             loss_weights[2] * b_loss)
    
    return total, (p_loss.item(), d_loss.item(), b_loss.item())


# Tổng loss (legacy - để tương thích với code cũ sử dụng fourier và complexity)
def total_loss(prob_pred, dist_pred, fourier_pred, complexity_pred, 
               prob_gt, dist_mask_gt, fourier_gt,
               loss_weights=(1.0, 0.2, 0.2, 0.1)):
    """
    Tổng loss = w1*prob + w2*dist + w3*fourier + w4*complexity
    fourier_gt: (B, 2*(n_harmonics+1) + 1, H, W) -> kênh cuối là complexity
    """
    prob_loss_fn = masked_bce_loss()
    
    mask = dist_mask_gt[:, -1:] 
    dist_gt = dist_mask_gt[:, :-1]

    comp_gt = fourier_gt[:, -1:]
    f_gt = fourier_gt[:, :-1]

    dist_loss_fn = masked_mae_loss(mask=mask)
    fourier_loss_fn = masked_fourier_loss(mask=mask)
    comp_loss_fn = complexity_loss_fn(mask=mask)

    p_loss = prob_loss_fn(prob_gt, prob_pred)
    d_loss = dist_loss_fn(dist_gt, dist_pred)
    f_loss = fourier_loss_fn(f_gt, fourier_pred)
    c_loss = comp_loss_fn(comp_gt, complexity_pred)

    total = (loss_weights[0] * p_loss + 
             loss_weights[1] * d_loss + 
             loss_weights[2] * f_loss + 
             loss_weights[3] * c_loss)
    
    return total, (p_loss.item(), d_loss.item(), f_loss.item(), c_loss.item())

