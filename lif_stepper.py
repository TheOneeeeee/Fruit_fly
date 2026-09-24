import numpy as np
import scipy.sparse as sp

# The connectome (neuron IDs + synaptic weight matrix) is the same fixed
# data for every fly — only the per-fly dynamic state (v, g, refractory,
# delay queue) actually needs to be independent. Reloading the files from
# disk in every LIFBrain.__init__ was pure repeated I/O + npz decompression
# for data that never changes, which is the main cause of the spawn-in lag.
# Caching it here means only the first fly pays that cost.
_connectome_cache = {}

def _load_connectome(ids_path, weights_path):
    key = (ids_path, weights_path)
    cached = _connectome_cache.get(key)
    if cached is None:
        ids = np.load(ids_path)
        id_to_idx = {nid: i for i, nid in enumerate(ids)}
        W = sp.load_npz(weights_path).tocsr()
        cached = (ids, id_to_idx, W)
        _connectome_cache[key] = cached
    return cached


class LIFBrain:
    def __init__(self, ids_path="data/subgraph_ids.npy", weights_path="data/subgraph_weights.npz", dt=1.0):
        # Shared, read-only across every fly using the same files.
        self.ids, self.id_to_idx, self.W = _load_connectome(ids_path, weights_path)
        self.N = len(self.ids)
        self.dt = dt  # ms per step

        # exact constants from model.py's default_params
        self.v_0, self.v_rst, self.v_th = -52.0, -52.0, -45.0
        self.t_mbr, self.tau, self.t_rfc, self.t_dly = 20.0, 5.0, 2.2, 1.8
        self.w_syn, self.f_poi = 0.275, 250.0

        # Per-instance dynamic state — genuinely needs to be independent per fly.
        self.v = np.full(self.N, self.v_0, dtype=np.float32)
        self.g = np.zeros(self.N, dtype=np.float32)
        self.refractory = np.zeros(self.N, dtype=np.float32)

        delay_steps = max(1, round(self.t_dly / self.dt))
        self.delay_queue = [np.zeros(self.N, dtype=np.float32) for _ in range(delay_steps)]

    def step(self, poisson_drive=None):
        dt = self.dt
        self.v += dt / self.t_mbr * (self.v_0 - self.v + self.g)
        self.g += -dt / self.tau * self.g

        if poisson_drive:
            for nid, rate_hz in poisson_drive.items():
                idx = self.id_to_idx.get(nid)
                if idx is not None and self.refractory[idx] <= 0:
                    if np.random.rand() < (rate_hz * dt / 1000.0):
                        self.v[idx] += self.w_syn * self.f_poi

        incoming = self.delay_queue.pop(0)
        self.g += incoming
        self.delay_queue.append(np.zeros(self.N, dtype=np.float32))

        self.refractory = np.maximum(0, self.refractory - dt)
        can_spike = self.refractory <= 0
        spiked = can_spike & (self.v > self.v_th)

        if spiked.any():
            self.v[spiked] = self.v_rst
            self.g[spiked] = 0
            self.refractory[spiked] = self.t_rfc
            outgoing = self.W.dot(spiked.astype(np.float32)) * self.w_syn
            self.delay_queue[-1] += outgoing

        return spiked