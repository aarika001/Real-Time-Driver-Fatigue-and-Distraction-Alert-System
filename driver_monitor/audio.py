"""Alert sounds. Soft = visual only (per proposal); visual+audio and urgent get distinct tones."""
import numpy as np
import config

SR = 22050


class AudioAlert:
    def __init__(self, enabled: bool = True):
        self.muted = False
        self.ok = False
        self.last = {}
        if not enabled:
            return
        try:
            import pygame
            pygame.mixer.pre_init(SR, -16, 1, 512)
            pygame.mixer.init()
            self.pg = pygame
            self.sounds = {
                2: self._make([(880, 0.18), (0, 0.08), (880, 0.18)]),
                3: self._make([(1200, 0.15), (0, 0.05)] * 3),
            }
            self.ok = True
        except Exception as e:                       # no pygame / no audio device
            print(f"[audio] disabled ({e}); falling back to terminal bell")

    def _make(self, seq):
        chunks = []
        for f, d in seq:
            tt = np.arange(int(SR * d)) / SR
            env = np.clip(np.minimum(tt, d - tt) / 0.01, 0, 1)
            chunks.append((0.4 * np.sin(2 * np.pi * f * tt) if f else np.zeros_like(tt)) * env)
        return self.pg.sndarray.make_sound((np.concatenate(chunks) * 32767).astype(np.int16))

    def handle(self, decision, t: float):
        a = decision.action_idx
        if self.muted or decision.degraded or a < 2:
            return
        if t - self.last.get(a, -1e9) < config.AUDIO_COOLDOWN_S[a]:
            return
        self.last[a] = t
        if self.ok:
            self.sounds[a].play()
        else:
            print("\a", end="", flush=True)
