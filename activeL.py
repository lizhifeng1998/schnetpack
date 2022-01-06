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
parser.add_argument('--learning_rate', type=float, default=1e-3, help='1e-3')
parser.add_argument('--lr_patience', type=int, default=5, help='5')
parser.add_argument('--activeL', type=int, default=10, help='10')
parser.add_argument('--dataset', default='6a_capped.db')
parser.add_argument('--test', action='store_true', default=False,
                    help='test only')
parser.add_argument('--n_atom_basis', type=int, default=64, help='64')
parser.add_argument('--n_filters', type=int, default=64, help='64')
parser.add_argument('--n_gaussians', type=int, default=100, help='100')
parser.add_argument('--n_interactions', type=int, default=3, help='3')
parser.add_argument('--cutoff', type=float, default=4., help='4.')
parser.add_argument('--final_epochs', type=int, default=-1,)
parser.add_argument('--emin', type=float, default=47113.71)
parser.add_argument('--noactive1', action='store_true', default=False, 'random choose')
parser.add_argument('--noactive2', action='store_true', default=False, 'choose after know all energy')
parser.add_argument('--active1', action='store_true', default=False)
args = parser.parse_args()

metadata = {'atomrefs': [[0.0], [-13.613121720568273], [0.0], [0.0], [0.0], [0.0], [-1029.8631226682135], [-1485.3025123714042], [-2042.6112359256108], [-2713.4848558896506], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]], 'atref_labels': ['energy']}
new_dataset = AtomsData(args.dataset, available_properties=['energy'])
new_dataset.set_metadata(metadata=metadata)
idx = np.random.permutation(len(new_dataset))
train_idx = idx[:args.num_train].tolist()
val_idx = idx[-args.num_val:].tolist()
test_idx = idx[args.num_train:-args.num_val].tolist()
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
    trn.TestHook(test_loader, rootpath+'/a0', every_n_epochs=args.activeL,
                contributions='contributions' if args.contributions else None)
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
    if i == args.iterations - 1 and args.final_epochs > 0: n_epochs =  args.final_epochs
    trainer.train(device=device, n_epochs=n_epochs)
    best_model = torch.load(os.path.join(rootpath, 'a'+str(i)+'/best_model'))
    err = 0
    print(len(test_loader))
    plt.clf()
    x, y = [], []
    for count, batch in enumerate(test_loader):
        batch = {k: v.to(device) for k, v in batch.items()}
        pred = best_model(batch)
        tmp = torch.sum(torch.abs(pred['energy']-batch['energy']))
        tmp = tmp.detach().cpu().numpy()
        err += tmp
        x += [w[0] for w in (batch['energy'].detach().cpu().numpy()+args.emin)*23.04]
        y += [w[0] for w in (pred['energy'].detach().cpu().numpy()+args.emin)*23.04]
        percent = '{:3.2f}'.format(count/len(test_loader)*100)
        print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")
    del batch, pred
    plt.plot(x,y,'.')
    plt.plot(x,x,'-')
    plt.savefig(rootpath+'/a'+str(i)+'/pred.png')
    with open(rootpath+'/a'+str(i)+'/pred.txt','w') as f:
        for j in range(len(x)): f.write(str(x[j])+' '+str(y[j])+'\n')
    err /= len(test)
    print('Test MAE', np.round(err, 2), 'eV =',
      np.round(err / (kcal/mol), 2), 'kcal/mol')
    with open(rootpath+'/test.txt','a') as f: f.write(str(err / (kcal/mol))+'\n')
    #
    if args.contributions:
        atom_var = np.var(hooks[-1].result1,axis=0)
        var = np.max(atom_var,axis=1)
        with open(rootpath+'/a'+str(i)+'/atom_var.txt','w') as f:
            for j in range(atom_var.shape[0]):
                for k in range(atom_var.shape[1]): f.write(str(atom_var[j][k][0])+' ')
                f.write('\n')
    else:
        var = np.var(hooks[-1].result1,axis=0)
    order = np.argsort(-var,axis=0)
    with open(rootpath+'/a'+str(i)+'/var.txt','w') as f:
        for j in range(order.shape[0]): f.write(str(var[j][0])+' '+str(order[j][0])+'\n')
    if args.noactive1:
        order = np.random.permutation(order.shape[0]).reshape(order.shape[0],1)
    elif args.noactive2:
        err = np.abs([y[j] - x[j] for j in range(len(x))])
        order = np.argsort(-err).reshape(len(x),1)
    new_idx = [test_idx[order[j][0]] for j in range(args.num_train)]
    test_idx = [test_idx[order[j][0]] for j in range(args.num_train,len(test_idx))]
    train_idx += new_idx
    with open(rootpath+'/a'+str(i)+'/test.idx','w') as f:
        for j in range(len(test_idx)): f.write(str(test_idx[j])+'\n')
    with open(rootpath+'/a'+str(i)+'/train.idx','w') as f:
        for j in range(len(train_idx)): f.write(str(train_idx[j])+'\n')
    del train, test, train_loader, test_loader
    train = create_subset(new_dataset, train_idx)
    test = create_subset(new_dataset, test_idx)
    train_loader = spk.AtomsLoader(train, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    test_loader = spk.AtomsLoader(test, batch_size=args.batch_size)
    means, stddevs = train_loader.get_statistics(
    'energy', divide_by_atoms=True, single_atom_ref=atomrefs)
    print('Mean atomization energy / atom:', means['energy'])
    print('Std. dev. atomization energy / atom:', stddevs['energy'])
    del trainer, model, optimizer
    if True:
        del schnet, output_U0
        schnet = spk.representation.SchNet(
        n_atom_basis=args.n_atom_basis, n_filters=args.n_filters,
        n_gaussians=args.n_gaussians, n_interactions=args.n_interactions,
        cutoff=args.cutoff, cutoff_network=spk.nn.cutoff.CosineCutoff
        )
        output_U0 = spk.atomistic.Atomwise(n_in=args.n_atom_basis, atomref=atomrefs['energy'], property='energy',
                                   mean=means['energy'], stddev=stddevs['energy'])
        model = spk.AtomisticModel(representation=schnet, output_modules=output_U0)
    else:
        model = best_model
        del model.output_modules[0].standardize
        model.output_modules[0].standardize = spk.nn.base.ScaleShift(means['energy'], stddevs['energy'])
    optimizer = Adam(model.parameters(), lr=args.learning_rate)
    del hooks
    hooks = [
    trn.CSVHook(log_path=rootpath, metrics=metrics),
    trn.ReduceLROnPlateauHook(
        optimizer,
        patience=args.lr_patience, factor=0.8, min_lr=1e-6,
        stop_after_min=True
    ),
    trn.PrintHook()]
    if i != args.iterations - 1:
        hooks.append(trn.TestHook(test_loader, rootpath+'/a'+str(i+1), every_n_epochs=args.activeL, contributions='contributions' if args.contributions else None))
    trainer = trn.Trainer(
    model_path=rootpath+'/a'+str(i+1),
    model=model,
    hooks=hooks,
    loss_fn=loss,
    optimizer=optimizer,
    train_loader=train_loader,
    validation_loader=val_loader,
    )
