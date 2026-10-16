"""Full-grid diagnostics from the corrected benchmark evaluator."""
import torch


def metrics(pred, ref):
    # Match the implementation's unnormalised directional rFFTs, including DC.
    def spatial(a):
        return .5 * (torch.fft.rfft(a, dim=0).abs().square().mean((1, 2)) +
                     torch.fft.rfft(a, dim=1).abs().square().mean((0, 2)))

    def temporal(a):
        return torch.fft.rfft(a, dim=2).abs().square().mean((0, 1))

    def log_error(a, b):
        return float((a.clamp_min(1e-12).log10() -
                      b.clamp_min(1e-12).log10()).square().mean())

    p, r = pred.double(), ref.double()
    vp, vr = p.var(correction=0), r.var(correction=0)
    kp = float((p - p.mean()).pow(4).mean() / vp.square() - 3)
    kr = float((r - r.mean()).pow(4).mean() / vr.square() - 3)
    ratio = float(vp / vr)
    return dict(var_ratio=ratio, abs_var_error=abs(ratio - 1),
        log_sp_mse=log_error(spatial(pred), spatial(ref)),
        log_tp_mse=log_error(temporal(pred), temporal(ref)),
        rel_l2=float((p-r).norm() / r.norm()), kurt_pred=kp, kurt_ref=kr,
        kurt_err=abs(kp-kr), mean_pred=float(p.mean()), mean_ref=float(r.mean()),
        var_pred=float(vp), var_ref=float(vr),
        negative_pred=float((p < 0).double().mean()),
        negative_ref=float((r < 0).double().mean()))
