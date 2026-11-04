"""Graph head on superpixels: an experimental border-aware refinement of the model's output.

Pixels are grouped into superpixels (SLIC on the yearly median radar, where speckle is averaged out and real borders
remain). Each superpixel is a node. Two nodes are linked if they touch; the link is weak when their radar looks
different, so information does not flow across a real border. A small graph network (2 layers of message passing, plain
PyTorch, no extra package) takes the frozen U-Net's mean class probabilities per node, adds what its neighbours say, and
outputs refined class scores per node. Every pixel takes its node's class.

Honest expectations: a U-Net already does message passing on a regular grid. A graph helps only where the links are not
a grid (object outlines, later also links across months). The first thing to test is `object_vote`, which has no
training at all: the mean probability per superpixel. If that already moves boundary F1, the learned head can add a little.

Train the head on VALIDATION regions' probabilities (the U-Net is overconfident on its own training regions), judge it on
test regions. Use `sharpness.passes_border_rule` as the decision rule, fixed in advance.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

IGNORE = 255


def superpixels(guide: np.ndarray, n_segments: int = 3000, compactness: float = 0.1) -> np.ndarray:
    """SLIC on a guide image (G,H,W) in dB (NaN allowed) -> int32 segment ids 0..N-1, shape (H,W)."""
    from skimage.segmentation import slic
    g = np.array(guide, np.float32)
    for b in range(g.shape[0]):
        bad = ~np.isfinite(g[b])
        g[b][bad] = np.nanmedian(g[b]) if (~bad).any() else 0.0
    lo, hi = g.reshape(g.shape[0], -1).min(1), g.reshape(g.shape[0], -1).max(1)
    g = (g - lo[:, None, None]) / np.maximum(hi - lo, 1e-6)[:, None, None]
    seg = slic(np.moveaxis(g, 0, -1), n_segments=n_segments, compactness=compactness, channel_axis=-1, start_label=0, enforce_connectivity=True)
    # relabel to 0..N-1 without gaps
    _, inv = np.unique(seg, return_inverse=True)
    return inv.reshape(seg.shape).astype(np.int32)


def node_means(seg: np.ndarray, values: np.ndarray) -> np.ndarray:
    """Mean of values (K,H,W) over each segment -> (N,K). NaN values are ignored."""
    n = int(seg.max()) + 1
    v = np.nan_to_num(values.astype(np.float64), nan=0.0).reshape(values.shape[0], -1)
    ok = np.isfinite(values.astype(np.float64)).reshape(values.shape[0], -1)
    s = seg.ravel()
    sums = np.stack([np.bincount(s, weights=v[k], minlength=n) for k in range(v.shape[0])], 1)
    cnt = np.stack([np.bincount(s, weights=ok[k].astype(float), minlength=n) for k in range(v.shape[0])], 1)
    return sums / np.maximum(cnt, 1)


def node_labels(seg: np.ndarray, label: np.ndarray, n_classes: int, ignore: int = IGNORE) -> np.ndarray:
    """Majority valid label per segment, ignore (255) where the segment has no valid label."""
    n = int(seg.max()) + 1
    ok = label != ignore
    idx = seg[ok].astype(np.int64) * n_classes + label[ok].astype(np.int64)
    counts = np.bincount(idx, minlength=n * n_classes).reshape(n, n_classes)
    out = counts.argmax(1).astype(np.uint8)
    out[counts.sum(1) == 0] = ignore
    return out


def adjacency_pairs(seg: np.ndarray) -> np.ndarray:
    """Unique undirected pairs (i,j), i<j, of segments that touch (4-neighbourhood). Shape (E,2)."""
    pairs = []
    for a, b in ((seg[:, :-1], seg[:, 1:]), (seg[:-1, :], seg[1:, :])):
        m = a != b
        lo, hi = np.minimum(a[m], b[m]), np.maximum(a[m], b[m])
        pairs.append(np.stack([lo, hi], 1))
    p = np.concatenate(pairs) if pairs else np.zeros((0, 2), np.int64)
    return np.unique(p, axis=0) if len(p) else p


def build_graph(seg: np.ndarray, probs: np.ndarray, guide: np.ndarray, theta: float = 1.5) -> dict:
    """Node features and a row-normalised adjacency (with self loops) as torch tensors.

    probs (C,H,W) mean softmax of the frozen model (NaN where no radar); guide (G,H,W) dB.
    Edge weight = exp(-d^2 / (2 theta^2)) with d = distance between the two nodes' mean guide values (dB)."""
    n = int(seg.max()) + 1
    p = node_means(seg, probs)                                   # (N,C)
    p = p / np.maximum(p.sum(1, keepdims=True), 1e-6)
    gm = node_means(seg, guide)                                  # (N,G)
    gs = np.sqrt(np.maximum(node_means(seg, guide ** 2) - gm ** 2, 0))
    size = np.bincount(seg.ravel(), minlength=n).astype(np.float64)
    x = np.concatenate([p, gm / 10.0, gs / 5.0, np.log1p(size)[:, None] / 5.0], 1).astype(np.float32)
    A = np.eye(n, dtype=np.float32)
    e = adjacency_pairs(seg)
    if len(e):
        d2 = ((gm[e[:, 0]] - gm[e[:, 1]]) ** 2).sum(1)
        w = np.exp(-d2 / (2 * theta ** 2)).astype(np.float32)
        A[e[:, 0], e[:, 1]] = w
        A[e[:, 1], e[:, 0]] = w
    A = A / A.sum(1, keepdims=True)
    return {"x": torch.from_numpy(x), "A": torch.from_numpy(A), "p": torch.from_numpy(p.astype(np.float32)), "seg": seg}


def object_vote(graph: dict) -> np.ndarray:
    """No training: the class with the highest mean probability inside each superpixel. Returns (N,) uint8."""
    return graph["p"].argmax(1).numpy().astype(np.uint8)


def pixels_from_nodes(seg: np.ndarray, node_class: np.ndarray) -> np.ndarray:
    return node_class[seg]


class GraphHead(nn.Module):
    """2 rounds of message passing; the output is log(mean probs) + a learned correction (starts at zero)."""

    def __init__(self, in_dim: int, n_classes: int, hidden: int = 32):
        super().__init__()
        self.l1 = nn.Linear(in_dim, hidden)
        self.l2 = nn.Linear(hidden, hidden)
        self.out = nn.Linear(hidden, n_classes)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, x, A, p):
        h = F.relu(A @ self.l1(x))
        h = F.relu(A @ self.l2(h))
        return torch.log(p.clamp_min(1e-6)) + self.out(A @ h)


def train_head(graphs: list[dict], labels: list[np.ndarray], n_classes: int, epochs: int = 150, lr: float = 5e-3,
               weight_decay: float = 1e-3, seed: int = 0) -> GraphHead:
    """graphs and per-node labels (255 = ignore) of the TRAINING regions. Full-batch Adam, fixed seed."""
    torch.manual_seed(seed)
    model = GraphHead(graphs[0]["x"].shape[1], n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    ys = [torch.from_numpy(l.astype(np.int64)) for l in labels]
    for _ in range(epochs):
        opt.zero_grad()
        loss = 0.0
        for g, y in zip(graphs, ys):
            loss = loss + F.cross_entropy(model(g["x"], g["A"], g["p"]), y, ignore_index=IGNORE)
        (loss / len(graphs)).backward()
        opt.step()
    return model


@torch.no_grad()
def predict_nodes(model: GraphHead, graph: dict) -> np.ndarray:
    model.eval()
    return model(graph["x"], graph["A"], graph["p"]).argmax(1).numpy().astype(np.uint8)
