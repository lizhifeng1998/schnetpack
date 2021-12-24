import torch
from ase.units import kcal, mol
import matplotlib.pyplot as plt
import numpy as np
import schnetpack.train as trn
from torch.optim import Adam
from schnetpack import AtomsData
import os
import schnetpack as spk
import argparse

parser = argparse.ArgumentParser(description='schnetpack tutorial-1')
parser.add_argument('--num_train', type=int, default=100,)
parser.add_argument('--num_val', type=int, default=500,)
parser.add_argument('--batch_size', type=int, default=100,)
args = parser.parse_args()

metadata = {'atomrefs': [[0.0], [-13.613121720568273], [0.0], [0.0], [0.0], [0.0], [-1029.8631226682135], [-1485.3025123714042], [-2042.6112359256108], [-2713.4848558896506], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], 'atref_labels': ['energy']}
new_dataset = AtomsData(args.dataset, available_properties=['energy'])
new_dataset.set_metadata(metadata=metadata)
idx = np.random.permutation(len(new_dataset))
train_idx = idx[:args.num_train].tolist()
val_idx = idx[-args.num_val:]
train = create_subset(new_dataset, train_idx)
val = create_subset(new_dataset, val_idx)
train_loader = spk.AtomsLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = spk.AtomsLoader(val, batch_size=args.batch_size)
