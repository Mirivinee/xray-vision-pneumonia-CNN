import os
import shutil

import numpy as np
from PIL import Image


def make_fake_dataset(root, n_train=60, n_val=4, n_test=14, seed=0):
    rng = np.random.default_rng(seed)

    def img(label, size=(96, 80)):
        a = rng.normal(100, 25, (size[1], size[0])).clip(0, 255)
        if label:
            a[25:60, 15:65] += 40
        return Image.fromarray(a.astype("uint8"))

    pid = 0
    first_train = None
    for split, n in (("train", n_train), ("val", n_val), ("test", n_test)):
        for cname, label in (("NORMAL", 0), ("PNEUMONIA", 1)):
            d = root / split / cname
            d.mkdir(parents=True, exist_ok=True)
            for k in range(n):
                pid += 1
                if label == 0:
                    name = f"IM-{pid:04d}-0001.jpeg" if k % 3 else f"NORMAL2-IM-{pid:04d}-0001.jpeg"
                else:
                    name = f"person{pid}_{'bacteria' if k % 2 else 'virus'}_{k}.jpeg"
                img(label).save(d / name, quality=95)
                if split == "train" and label == 1 and first_train is None:
                    first_train = d / name
    shutil.copy(first_train, root / "test" / "PNEUMONIA" / "person9999_bacteria_1.jpeg")
    extra = root / "train" / "NORMAL" / "IM-8888-0001.jpeg"
    shutil.copy(root / "train" / "NORMAL" / sorted(os.listdir(root / "train" / "NORMAL"))[0], extra)
    shutil.copy(extra, root / "train" / "NORMAL" / "IM-8889-0001.jpeg")
