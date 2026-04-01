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


#Tổng loss
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
