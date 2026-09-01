import random
from collections.abc import Sequence


def split_sample_ids(
    sample_ids: Sequence[int], *, seed :int = 420, train_frac:float = 0.7,val_frac: float = 0.15) -> dict[str, list[str]]:

    if train_frac < 0 or train_frac > 1:
        raise ValueError(f"train_frac must be between 0 and 1, but got {train_frac}")
    if val_frac < 0 or val_frac > 1:
        raise ValueError(f"val_frac must be between 0 and 1, but got {val_frac}")

    if train_frac + val_frac >= 1:
        raise ValueError("The train fraction and val fraction exceeds 1!")

    ids = sorted(str(sample_id) for sample_id in sample_ids)

    if len(ids) != len(set(ids)):
        raise ValueError("sample_ids contains duplicates")

    rng = random.Random(seed)
    rng.shuffle(ids)

    n_train = int(len(ids) * train_frac)
    n_val = int(len(ids) * val_frac)

    train_ids = sorted(ids[:n_train])
    val_ids = sorted(ids[n_train: n_train + n_val])
    test_ids = sorted(ids[n_train + n_val:])

    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids,
    }



