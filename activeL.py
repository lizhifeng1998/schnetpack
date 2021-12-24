from schnetpack.data.partitioning import create_subset
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
from torch import nn
def mse_loss(batch, result):
    diff = batch['energy']-result['energy']
    err_sq = torch.mean(diff ** 2)
    return err_sq

parser = argparse.ArgumentParser(description='schnetpack tutorial-1')
parser.add_argument('--num_train', type=int, default=100,)
parser.add_argument('--num_val', type=int, default=500,)
parser.add_argument('--batch_size', type=int, default=100,)
parser.add_argument('--epochs', type=int, default=200, help='200')
parser.add_argument('--iterations', type=int, default=5, help='5')
parser.add_argument('--learning_rate', type=float, default=1e-2, help='1e-2')
parser.add_argument('--lr_patience', type=int, default=5, help='5')
parser.add_argument('--activeL', type=int, default=10, help='10')
args = parser.parse_args()

metadata = {'atomrefs': [[0.0], [-13.613121720568273], [0.0], [0.0], [0.0], [0.0], [-1029.8631226682135], [-1485.3025123714042], [-2042.6112359256108], [-2713.4848558896506], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], 'atref_labels': ['energy']}
new_dataset = AtomsData(args.dataset, available_properties=['energy'])
new_dataset.set_metadata(metadata=metadata)
idx = np.random.permutation(len(new_dataset))
train_idx = idx[:args.num_train].tolist()
val_idx = idx[-args.num_val:]
test_idx = idx[args.num_train:-args.num_val]
train = create_subset(new_dataset, train_idx)
val = create_subset(new_dataset, val_idx)
test = create_subset(new_dataset, test_idx)
train_loader = spk.AtomsLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
val_loader = spk.AtomsLoader(val, batch_size=args.batch_size)
test_loader = spk.AtomsLoader(test, batch_size=args.batch_size)

atomrefs = new_dataset.get_atomref('energy')
print('U0 of hyrogen:', '{:.2f}'.format(atomrefs['energy'][1][0]), 'eV')
print('U0 of carbon:', '{:.2f}'.format(atomrefs['energy'][6][0]), 'eV')
print('U0 of oxygen:', '{:.2f}'.format(atomrefs['energy'][8][0]), 'eV')

means, stddevs = train_loader.get_statistics(
    'energy', divide_by_atoms=True, single_atom_ref=atomrefs
)
print('Mean atomization energy / atom:', means['energy'])
print('Std. dev. atomization energy / atom:', stddevs['energy'])

schnet = spk.representation.SchNet(
    n_atom_basis=args.n_atom_basis, n_filters=args.n_filters,
    n_gaussians=args.n_gaussians, n_interactions=args.n_interactions,
    cutoff=args.cutoff, cutoff_network=spk.nn.cutoff.CosineCutoff
)

output_U0 = spk.atomistic.Atomwise(n_in=args.n_atom_basis, atomref=atomrefs['energy'], property='energy',
                                   mean=means['energy'], stddev=stddevs['energy'])
model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)

optimizer = Adam(model.parameters(), lr=args.learning_rate)
loss = trn.build_mse_loss(['energy'])

metrics = [spk.metrics.MeanAbsoluteError('energy')]
rootpath = './activeL'
if not os.path.exists(rootpath):
    os.makedirs(rootpath)
hooks = [
    trn.CSVHook(log_path=rootpath, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=args.lr_patience, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    ),
    trn.PrintHook(),
    trn.TestHook(test_loader, every_n_epochs=args.activeL)
]

trainer = trn.Trainer(
    model_path=rootpath+'/a0',
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

device = "cuda"
n_epochs = args.epochs

for i in range(args.iterations):
    trainer.train(device=device, n_epochs=n_epochs)
    var = np.var(hooks[-1].result,axis=0)
    order = np.argsort(var,axis=0)
    new_idx = np.array([test_idx[order[i][0]] for i in range(args.num_train)])
    test_idx = np.array([test_idx[order[i][0]] for i in range(args.num_train,len(test_idx))])
    del train, test, train_loader, test_loader
    train = create_subset(new_dataset, np.concatenate((train_idx,new_idx)))
    test = create_subset(new_dataset, test_idx)
    train_loader = spk.AtomsLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = spk.AtomsLoader(test, batch_size=args.batch_size)
    means, stddevs = train_loader.get_statistics(
    'energy', divide_by_atoms=True, single_atom_ref=atomrefs)
    print('Mean atomization energy / atom:', means['energy'])
    print('Std. dev. atomization energy / atom:', stddevs['energy'])
    del trainer, model, optimizer, schnet, output_U0
    schnet = spk.representation.SchNet(
    n_atom_basis=args.n_atom_basis, n_filters=args.n_filters,
    n_gaussians=args.n_gaussians, n_interactions=args.n_interactions,
    cutoff=args.cutoff, cutoff_network=spk.nn.cutoff.CosineCutoff
    )
    output_U0 = spk.atomistic.Atomwise(n_in=args.n_atom_basis, atomref=atomrefs['energy'], property='energy',
                                   mean=means['energy'], stddev=stddevs['energy'])
    model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    hooks[-1] = trn.TestHook(test_loader, every_n_epochs=args.activeL)
    trainer = trn.Trainer(
    model_path=rootpath+'/'+str(i+1),
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    )