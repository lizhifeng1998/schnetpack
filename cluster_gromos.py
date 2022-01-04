import torch
from ase.units import kcal, mol
import matplotlib.pyplot as plt
import numpy as np
import schnetpack.train as trn
from torch.optim import Adam
from schnetpack import AtomsData
import os
import sys
import schnetpack as spk
import argparse
from schnetpack.data.partitioning import create_subset

parser = argparse.ArgumentParser(description='schnetpack tutorial-1')
parser.add_argument('--test', action='store_true', default=False,
                    help='test only')
parser.add_argument('--num_train', type=int, default=1000, )
parser.add_argument('--num_val', type=int, default=500, help='500')
parser.add_argument('--epochs', type=int, default=200, help='200')
parser.add_argument('--n_atom_basis', type=int, default=64, help='64')
parser.add_argument('--n_filters', type=int, default=64, help='64')
parser.add_argument('--n_gaussians', type=int, default=100, help='100')
parser.add_argument('--n_interactions', type=int, default=3, help='3')
parser.add_argument('--cutoff', type=float, default=4., help='4')
parser.add_argument('--learning_rate', type=float, default=1e-3, help='1e-3')
parser.add_argument('--batch_size', type=int, default=100, help='100')
parser.add_argument('--dataset', default='6a_capped.db')
parser.add_argument('--emin', type=float, default=47113.71)
parser.add_argument('--key_out', default='energy')
parser.add_argument('--train')
args = parser.parse_args()

metadata = {'atomrefs': [[0.0], [-13.613121720568273], [0.0], [0.0], [0.0], [0.0], [-1029.8631226682135], [-1485.3025123714042], [-2042.6112359256108], [-2713.4848558896506], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], 'atref_labels': ['energy']}

rootpath = './clusterG'
if not os.path.exists(rootpath):
    os.makedirs(rootpath)
with open(rootpath+'/cmd','a') as f:
    for x in sys.argv: f.write(x+' ')
    f.write(str(sys.argv)+'\n')

new_dataset = AtomsData(args.dataset, available_properties=['energy'])
new_dataset.set_metadata(metadata=metadata)

train_idx = []
with open(args.train,'r') as f:
    for x in f.readlines(): train_idx.append(int(x.strip().split()[0]))
idx = np.random.permutation(len(new_dataset))
val_idx, test_idx = [], []
i=0
for x in idx:
    if x in train_idx: continue
    if i < args.num_val: val_idx.append(int(x))
    else: test_idx.append(int(x))
    i += 1
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

def mse_loss(batch, result):
    diff = batch['energy']-result['energy']
    err_sq = torch.mean(diff ** 2)
    return err_sq

optimizer = Adam(model.parameters(), lr=args.learning_rate)

loss = trn.build_mse_loss(['energy'])

metrics = [spk.metrics.MeanAbsoluteError('energy')]
hooks = [
    trn.CSVHook(log_path=rootpath, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=5, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    ),
    trn.PrintHook()
]
if not args.test:
    trainer = trn.Trainer(
    model_path=rootpath,
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

results = np.loadtxt(os.path.join(rootpath, 'log.csv'),
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
plt.savefig('mytut.png')

best_model = torch.load(os.path.join(rootpath, 'best_model'))

test_loader = spk.AtomsLoader(test, batch_size=args.batch_size)

err = 0
print(len(test_loader))
plt.clf()
x, y = [], []
for count, batch in enumerate(test_loader):
    # move batch to GPU, if necessary
    batch = {k: v.to(device) for k, v in batch.items()}

    # apply model
    pred = best_model(batch)

    # calculate absolute error
    tmp = torch.sum(torch.abs(pred[args.key_out]-batch['energy']))
    tmp = tmp.detach().cpu().numpy()  # detach from graph & convert to numpy
    err += tmp
    
    x += [w[0] for w in (batch['energy'].detach().cpu().numpy()+args.emin)*23.04]
    y += [w[0] for w in (pred[args.key_out].detach().cpu().numpy()+args.emin)*23.04]

    # log progress
    percent = '{:3.2f}'.format(count/len(test_loader)*100)
    print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")

plt.plot(x,y,'.')
plt.plot(x,x,'-')
plt.savefig(rootpath+'/test.png')
with open(rootpath+'/result.txt','w') as f:
    for i in range(len(x)): f.write(str(x[i])+' '+str(y[i])+'\n')
err /= len(test)
print('Test MAE', np.round(err, 2), 'eV =',
      np.round(err / (kcal/mol), 2), 'kcal/mol')
