import json
import numpy as np


def label_validator():
    with open('train.json') as f:
        annots = json.load(f)["annotations"]

    set_ids = set()
    for an in annots:
        ids = an["labelId"]
        ids = [int(id) for id in ids]
        set_ids.update(ids)

    labels = np.array(list(set_ids))
    max_label = np.max(labels)
    min_label = np.min(labels)

    validate = True
    for label in range(min_label, max_label+1):
        if label not in labels:
            validate = False
            break

    print("validation", validate)


