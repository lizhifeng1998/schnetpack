import torch
from ase.units import kcal, mol
import matplotlib.pyplot as plt
import numpy as np
import schnetpack.train as trn
from torch.optim import Adam
from schnetpack.datasets import QM9
import os
import schnetpack as spk
import argparse

parser = argparse.ArgumentParser(description='schnetpack tutorial-2')
parser.add_argument('--test', action='store_true', default=False,
                    help='test only')
parser.add_argument('--num_train', type=int, default=1000,)
parser.add_argument('--num_val', type=int, default=500,)
parser.add_argument('--epochs', type=int, default=200,)
args = parser.parse_args()

qm9tut = './qm9tut'
if not os.path.exists('qm9tut'):
    os.makedirs(qm9tut)

qm9data = QM9('./qm9.db', download=True,
              load_only=[QM9.U0], remove_uncharacterized=True)

train, val, test = spk.train_test_split(
    data=qm9data,
    num_train=args.num_train,
    num_val=args.num_val,
    split_file=os.path.join(qm9tut, "split.npz"),
)

train_loader = spk.AtomsLoader(train, batch_size=100, shuffle=True, pin_memory=True)
val_loader = spk.AtomsLoader(val, batch_size=100)

atomrefs = qm9data.get_atomref(QM9.U0)
print('U0 of hyrogen:', '{:.2f}'.format(atomrefs[QM9.U0][1][0]), 'eV')
print('U0 of carbon:', '{:.2f}'.format(atomrefs[QM9.U0][6][0]), 'eV')
print('U0 of oxygen:', '{:.2f}'.format(atomrefs[QM9.U0][8][0]), 'eV')

means, stddevs = train_loader.get_statistics(
    QM9.U0, divide_by_atoms=True, single_atom_ref=atomrefs
)
print('Mean atomization energy / atom:', means[QM9.U0])
print('Std. dev. atomization energy / atom:', stddevs[QM9.U0])

schnet = spk.representation.SchNet(
    n_atom_basis=30, n_filters=30, n_gaussians=20, n_interactions=5,
    cutoff=4., cutoff_network=spk.nn.cutoff.CosineCutoff
)

output_U0 = spk.atomistic.Atomwise(n_in=30, atomref=atomrefs[QM9.U0], property=QM9.U0,
                                   mean=means[QM9.U0], stddev=stddevs[QM9.U0])
model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)


# loss function

def mse_loss(batch, result):
    diff = batch[QM9.U0]-result[QM9.U0]
    err_sq = torch.mean(diff ** 2)
    return err_sq


# build optimizer
optimizer = Adam(model.parameters(), lr=1e-2)

if not args.test:
    try:
        os.remove('./qm9tut/checkpoints')
        os.remove('./qm9tut/log.csv')
        os.removedirs('./qm9tut/log.csv')
        os.removedirs('./qm9tut/checkpoints')
    except:
        pass


loss = trn.build_mse_loss([QM9.U0])

metrics = [spk.metrics.MeanAbsoluteError(QM9.U0)]
hooks = [
    trn.CSVHook(log_path=qm9tut, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    ),
    trn.PrintHook()
]

trainer = trn.Trainer(
    model_path=qm9tut,
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
)

device = "cuda"  # change to 'cpu' if gpu is not available
n_epochs = args.epochs  # takes about 10 min on a notebook GPU. reduces for playing around
if not args.test:
    trainer.train(device=device, n_epochs=n_epochs)

results = np.loadtxt(os.path.join(qm9tut, 'log.csv'),
                     skiprows=1, delimiter=',')

time = results[:, 0]-results[0, 0]
learning_rate = results[:, 1]
train_loss = results[:, 2]
val_loss = results[:, 3]
val_mae = results[:, 4]

print('Final validation MAE:', np.round(val_mae[-1], 2), 'eV =',
      np.round(val_mae[-1] / (kcal/mol), 2), 'kcal/mol')

plt.figure(figsize=(14, 5))
plt.subplot(1, 2, 1)
plt.plot(time, val_loss, label='Validation')
plt.plot(time, train_loss, label='Train')
plt.yscale('log')
plt.ylabel('Loss [eV]')
plt.xlabel('Time [s]')
plt.legend()
plt.subplot(1, 2, 2)
plt.plot(time, val_mae)
plt.ylabel('mean abs. error [eV]')
plt.xlabel('Time [s]')
plt.savefig('qm9.png')

best_model = torch.load(os.path.join(qm9tut, 'best_model'))

test_loader = spk.AtomsLoader(test, batch_size=100)

err = 0
print(len(test_loader))
for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}

    # apply model
    pred = best_model(batch)

    # calculate absolute error
    tmp = torch.sum(torch.abs(pred[QM9.U0]-batch[QM9.U0]))
    tmp = tmp.detach().cpu().numpy()  # detach from graph & convert to numpy
    err += tmp

    # log progress
    percent = '{:3.2f}'.format(count/len(test_loader)*100)
    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

err /= len(test)
print('Test MAE', np.round(err, 2), 'eV =',
      np.round(err / (kcal/mol), 2), 'kcal/mol')
