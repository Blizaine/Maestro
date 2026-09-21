"""Differentiable Mothersuperior joint_v6 objectives, ported to Maestro's split model.

Reference: scripts/joint_v6.py at e2e63d859f3af879baf1b4d4e9f22d1eeda6fde5.
The AR weights and VAE stay frozen, but their input gradients must stay enabled.
"""
import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .protocol import MUSIC_END
from .training_math import ar_layer, acoustic_forward


def straight_through_embeddings(logits, embeddings):
    probabilities = logits.float().softmax(-1)
    indices = probabilities.argmax(-1)
    hard = F.one_hot(indices, logits.shape[-1]).float()
    return ((hard + probabilities - probabilities.detach()).to(embeddings.dtype) @ embeddings), indices


def soft_token_loss(logits, targets, neighbors, weights):
    keep = targets != -100
    logits, targets = logits[keep].float(), targets[keep]
    if not targets.numel():
        raise ValueError('The regularizer batch contains no valid tokens')
    logp = logits.log_softmax(-1)
    return (.75 * -logp.gather(1, targets[:, None])[:, 0]
            + .25 * -(logp.gather(1, neighbors[targets]) * weights[targets]).sum(1)).mean()


def pair_flow(ar, nar, prefix, codec_embeddings, target, t, noise, *, gradients):
    embed = ar.model.embed_tokens
    before = embed(torch.tensor(prefix, device=codec_embeddings.device))
    after = embed(torch.tensor([MUSIC_END], device=codec_embeddings.device))
    x = torch.cat((before, codec_embeddings.to(before.dtype), after))
    cache = []
    for layer in ar.model.layers:
        x, k, v = checkpoint(ar_layer, layer, x, use_reentrant=False) if gradients else ar_layer(layer, x)
        cache.append((k, v))
    mixed = t * noise + (1 - t) * target
    prediction = acoustic_forward(nar, mixed, torch.logit(torch.as_tensor(t, device=mixed.device)),
                                  cache, len(x), gradients=gradients)
    return F.mse_loss(prediction.float(), noise - target), prediction, mixed


class SpectralLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchaudio.transforms import MelSpectrogram
        self.mel = MelSpectrogram(48000, n_fft=2048, hop_length=480, n_mels=128, power=1.)
        for n in (512, 1024, 2048):
            self.register_buffer('window_' + str(n), torch.hann_window(n))

    def forward(self, prediction, reference):
        if prediction.shape != reference.shape or prediction.ndim != 2 or prediction.shape[0] != 2:
            raise ValueError('Waveform supervision requires aligned stereo samples')
        p, r = prediction.float().mean(0), reference.float().mean(0)
        mel = ((self.mel(p) + 1e-5).log() - (self.mel(r) + 1e-5).log()).abs().mean()
        mr = p.new_zeros(())
        for n in (512, 1024, 2048):
            window = getattr(self, 'window_' + str(n))
            a = torch.stft(p, n, hop_length=n // 4, window=window, return_complex=True).abs()
            b = torch.stft(r, n, hop_length=n // 4, window=window, return_complex=True).abs()
            mr = mr + ((a - b).norm() / (b.norm() + 1e-6)
                       + ((a + 1e-5).log() - (b + 1e-5).log()).abs().mean()) / 3
        def side(x):
            return (((x[0] - x[1]) / 2).square().mean() + 1e-7).log() - (((x[0] + x[1]) / 2).square().mean() + 1e-7).log()
        return mel + .5 * mr + .5 * (side(prediction.float()) - side(reference.float())).abs()


def waveform_loss(decoder, spectral, mixed, prediction, t, audio, start, *, frames=150, margin=25):
    # Short held-out excerpts can still be checked. Train windows remain 512.
    frames = min(frames, len(mixed) - 2 * margin)
    if frames < 25:
        raise ValueError('Use a longer check excerpt for waveform evaluation (at least 3 seconds)')
    left = (len(mixed) - frames) // 2 - margin
    clean = (mixed - t * prediction)[left:left + frames + 2 * margin]
    # Call the differentiable decoder, never the inference-mode decode wrapper.
    wav = checkpoint(decoder, clean.T[None].float(), use_reentrant=False)[0]
    wav = wav[:, margin * 1920:(margin + frames) * 1920]
    offset = (start + left + margin) * 1920
    ref = torch.as_tensor(audio[offset:offset + frames * 1920].T.copy(), device=wav.device)
    if ref.shape != wav.shape:
        raise ValueError('The waveform training crop is not aligned to its original recording')
    return spectral(wav, ref)
