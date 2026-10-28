import copy
import random

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from .common import fast_macro_f1


class MLP(nn.Module):
    def __init__(self, input_dim=21, hidden=(64, 32), dropout=0.0):
        super().__init__()
        layers = []
        for width in hidden:
            layers.extend([nn.Linear(input_dim, width), nn.ReLU()])
            if dropout:
                layers.append(nn.Dropout(dropout))
            input_dim = width
        layers.append(nn.Linear(input_dim, 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def classification_loss(logits, target, weights=None, gamma=0.0):
    # pt comes from unweighted CE. Alpha weights must not alter pt in focal loss.
    ce = F.cross_entropy(logits, target, reduction="none")
    loss = (1 - torch.exp(-ce)).pow(gamma) * ce
    if weights is not None:
        loss = loss * weights[target]
    return loss.mean()


def predict_neural(model, x):
    model.eval()
    with torch.inference_mode():
        tensor = torch.from_numpy(np.asarray(x, dtype=np.float32))
        return torch.cat([model(batch).softmax(dim=1) for batch in tensor.split(8192)]).numpy()


def train_neural(x, y, xv, yv, config, seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    model = MLP(x.shape[1], tuple(config["hidden"]), config.get("dropout", 0))
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config.get("weight_decay", 0))
    counts = np.bincount(y, minlength=3)
    power = config.get("class_weight_power", 0)
    weights = (len(y) / (3 * counts)) ** power
    # Average training-example weight is one, keeping learning-rate interpretation comparable.
    weights = weights / np.average(weights, weights=counts)
    weights = torch.tensor(weights, dtype=torch.float32)
    xt, yt = torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)
    best_score, best_epoch, best_state = -1, 0, None
    history = []
    generator = torch.Generator().manual_seed(seed)
    for epoch in range(1, config["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(y), generator=generator)
        total = 0.0
        for ids in permutation.split(config.get("batch_size", 256)):
            optimizer.zero_grad(set_to_none=True)
            loss = classification_loss(model(xt[ids]), yt[ids], weights, config.get("focal_gamma", 0))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
            optimizer.step()
            total += float(loss.detach()) * len(ids)
        pv = predict_neural(model, xv)
        macro = float(fast_macro_f1(yv, pv.argmax(axis=1)))
        # Report ordinary CE alongside weighted/focal training objective for interpretation.
        validation_ce = float(-np.log(np.clip(pv[np.arange(len(yv)), yv], 1e-12, 1)).mean())
        history.append({"epoch": epoch, "train_objective": total / len(y), "validation_ce": validation_ce, "validation_macro_f1": macro})
        print(f"{config['name']} Epoch [{epoch:02d}/{config['epochs']}] Train objective {total/len(y):.4f} | Validation CE {validation_ce:.4f} | Macro F1 {macro:.4f}", flush=True)
        if macro > best_score:
            best_score, best_epoch, best_state = macro, epoch, copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), {"best_epoch": best_epoch, "tuning_macro_f1_at_checkpoint": best_score,
                                           "class_weights": weights.tolist(), "device": "cpu"}
