# Slot Attention

Reproducing slot-attention based object discovery experiment.

## Training:

```bash
python3 train.py
```

Training progression plot for loss:

![loss](fig/2022-06-19-loss.png)

Learning rate scheduling visualization:

![lr](fig/2022-06-19-lr.png)

## Result visualization:

Below script shows the results for unsupervised object discovery:

```bash
python3 evaluate.py
```

![vis](fig/vis.png)

Accordingly, we find each slot that _sometimes_ successfully finds discrete objects.
