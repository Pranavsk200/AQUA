import numpy as np

from nn.layers import make_mlp, mask_list, masked_layers, save_model
from prune.masking import SelfPruner
from prune.schedule import cubic_plan, oneshot_plan
from train.loop import evaluate, fit
from train.optim import Adam

HIDDEN = (256, 128)
BATCH = 128
LR = 1e-3


def run_one(data, seed, criterion="taylor1", sparsity=0.9, epochs=20, schedule="cubic",
            regrow=0.0, normalize=True, min_density=0.01, ramp_start=2, ramp_end=0.75,
            every=100, ema=0.9, save_path=None, log=None):
    rng = np.random.default_rng(seed)
    sizes = [data["n_features"], *HIDDEN, data["n_classes"]]
    model = make_mlp(sizes, rng, masked=True)
    params = model.parameters()
    opt = Adam(params, lr=LR, masks=mask_list(model, params))

    pruner = None
    if sparsity > 0:
        spe = int(np.ceil(len(data["X_train"]) / BATCH))
        start, end = ramp_start * spe, int(ramp_end * epochs * spe)
        if schedule == "cubic":
            plan = cubic_plan(sparsity, start, end, every)
        else:
            plan = oneshot_plan(sparsity, end)
        pruner = SelfPruner(masked_layers(model), criterion, plan, seed=seed, regrow=regrow,
                            cooldown=2 * every, grace=2 * every, normalize=normalize,
                            min_density=min_density, ema=ema, optimizer=opt)

    history = fit(model, opt, data, epochs, BATCH, seed, pruner=pruner, log=log)
    test_loss, test_acc = evaluate(model, data["X_test"], data["y_test"])

    if save_path:
        save_model(model, save_path)

    layers = masked_layers(model)
    active = sum(int(l.mask.sum()) for l in layers)
    total = sum(l.mask.size for l in layers)
    biases = sum(l.bias.data.size for l in layers)
    return {
        "model": model,
        "history": history,
        "pruner": pruner,
        "row": {
            "criterion": criterion if sparsity > 0 else "dense",
            "target_sparsity": sparsity,
            "seed": seed,
            "achieved_sparsity": 1 - active / total,
            "sparsity_incl_bias": 1 - (active + biases) / (total + biases),
            "active_weights": active,
            "val_acc": history[-1]["val_acc"],
            "test_acc": test_acc,
            "test_loss": test_loss,
            "train_acc": history[-1]["train_acc"],
        },
    }
