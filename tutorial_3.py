import schnetpack as spk
import os
from schnetpack import AtomsData
from schnetpack.datasets import MD17
import torch
from torch.optim import Adam
import schnetpack.train as trn
import numpy as np
import matplotlib.pyplot as plt
from ase.units import kcal, mol
from ase import io
import argparse

parser = argparse.ArgumentParser(description='schnetpack tutorial-3')
parser.add_argument('--dataset', nargs='+')
parser.add_argument('--bs_train', type=int, default=24, help='train batch size 32')
parser.add_argument('--bs_val', type=int, default=24, help='val batch size 32')
args = parser.parse_args()

forcetut = './forcetut'
if not os.path.exists(forcetut):
    os.makedirs(forcetut)
dataset = AtomsData(args.dataset[0], available_properties=['forces','energy'])
# ethanol_data = MD17(os.path.join(forcetut,'ethanol.db'), molecule='ethanol')
# atoms, properties = ethanol_data.get_properties(0)
atoms, properties = dataset.get_properties(0)
print('Loaded properties:\n', *['{:s}\n'.format(i) for i in properties.keys()])
print('Forces:\n', properties[MD17.forces])
print('Shape:\n', properties[MD17.forces].shape)
train, val, test = spk.train_test_split(
        # data=ethanol_data,
        data=dataset,
        num_train=15000, #1000,
        num_val=5000, #500,
        split_file=os.path.join(forcetut, "split.npz"),
    )
# check if a GPU is available and use a CPU otherwise
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
if True:
    train_loader = spk.AtomsLoader(train, batch_size=args.bs_train, shuffle=True)
    val_loader = spk.AtomsLoader(val, batch_size=args.bs_val)
    means, stddevs = train_loader.get_statistics(
        spk.datasets.MD17.energy, divide_by_atoms=True
    )
    print('Mean atomization energy / atom:      {:12.4f} [kcal/mol]'.format(means[MD17.energy][0]))
    print('Std. dev. atomization energy / atom: {:12.4f} [kcal/mol]'.format(stddevs[MD17.energy][0]))
    n_features = 128
    schnet = spk.representation.SchNet(
        n_atom_basis=n_features,
        n_filters=n_features,
        n_gaussians=25,
        n_interactions=4,
        cutoff=5.,
        cutoff_network=spk.nn.cutoff.CosineCutoff
    )
    energy_model = spk.atomistic.Atomwise(
        n_in=n_features,
        property=MD17.energy,
        mean=means[MD17.energy],
        stddev=stddevs[MD17.energy],
        derivative=MD17.forces,
        negative_dr=True
    )
    model = spk.AtomisticModel(representation=schnet, output_modules=energy_model)
    # tradeoff
    rho_tradeoff = 0.1
    # loss function
    def loss(batch, result):
        # compute the mean squared error on the energies
        diff_energy = batch[MD17.energy]-result[MD17.energy]
        err_sq_energy = torch.mean(diff_energy ** 2)
        # compute the mean squared error on the forces
        diff_forces = batch[MD17.forces]-result[MD17.forces]
        err_sq_forces = torch.mean(diff_forces ** 2)
        # build the combined loss function
        err_sq = rho_tradeoff*err_sq_energy + (1-rho_tradeoff)*err_sq_forces
        return err_sq
    # build optimizer
    optimizer = Adam(model.parameters(), lr=5e-4)
    # before setting up the trainer, remove previous training checkpoints and logs
    try:
        os.remove('./forcetut/checkpoints')
    except:
        print('rm checkpoints failed')
    try:
        os.remove('./forcetut/log.csv')
    except:
        print('rm log.csv failed')
    # %rm -rf ./forcetut/checkpoints
    # %rm -rf ./forcetut/log.csv
    # set up metrics
    metrics = [
        spk.metrics.MeanAbsoluteError(MD17.energy),
        spk.metrics.MeanAbsoluteError(MD17.forces)
    ]
    # construct hooks
    hooks = [
        trn.CSVHook(log_path=forcetut, metrics=metrics), 
        trn.ReduceLROnPlateauHook(
            optimizer, 
            patience=5, factor=0.8, min_lr=1e-6,
            stop_after_min=True
        ),
        trn.PrintHook()
    ]
    trainer = trn.Trainer(
        model_path=forcetut,
        model=model,
        hooks=hooks,
        loss_fn=loss,
        optimizer=optimizer,
        train_loader=train_loader,
        validation_loader=val_loader,
    )
    # determine number of epochs and train
    n_epochs = 300
    trainer.train(device=device, n_epochs=n_epochs)
    # Load logged results
    results = np.loadtxt(os.path.join(forcetut, 'log.csv'), skiprows=1, delimiter=',')
    # Determine time axis
    time = results[:,0]-results[0,0]
    # Load the validation MAEs
    energy_mae = results[:,4]
    forces_mae = results[:,5]
    # Get final validation errors
    print('Validation MAE:')
    print('    energy: {:10.3f} kcal/mol'.format(energy_mae[-1]))
    print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_mae[-1]))
    # Construct figure
    plt.figure(figsize=(14,5))
    # Plot energies
    plt.subplot(1,2,1)
    plt.plot(time, energy_mae)
    plt.title('Energy')
    plt.ylabel('MAE [kcal/mol]')
    plt.xlabel('Time [s]')
    # Plot forces
    plt.subplot(1,2,2)
    plt.plot(time, forces_mae)
    plt.title('Forces')
    plt.ylabel('MAE [kcal/mol/\u212B]')
    plt.xlabel('Time [s]')
    plt.savefig(os.path.join(forcetut, 'MAE.png'))

if False:
    best_model = torch.load(os.path.join(forcetut, 'best_model'))
    test_loader = spk.AtomsLoader(test, batch_size=25)
    energy_error = 0.0
    forces_error = 0.0
    X, Y = torch.tensor([]), torch.tensor([])
    for count, batch in enumerate(test_loader):    
        # move batch to GPU, if necessary
        batch = {k: v.to(device) for k, v in batch.items()}
        # apply model
        pred = best_model(batch)
        X = torch.cat((X,batch[MD17.energy].detach().cpu()))
        Y = torch.cat((Y,pred[MD17.energy].detach().cpu()))
        # calculate absolute error of energies
        tmp_energy = torch.sum(torch.abs(pred[MD17.energy] - batch[MD17.energy]))
        tmp_energy = tmp_energy.detach().cpu().numpy() # detach from graph & convert to numpy
        energy_error += tmp_energy
        # calculate absolute error of forces, where we compute the mean over the n_atoms x 3 dimensions
        tmp_forces = torch.sum(
            torch.mean(torch.abs(pred[MD17.forces] - batch[MD17.forces]), dim=(1,2))
        )
        tmp_forces = tmp_forces.detach().cpu().numpy() # detach from graph & convert to numpy
        forces_error += tmp_forces
        # log progress
        percent = '{:3.2f}'.format(count/len(test_loader)*100)
        print('Progress:', percent+'%'+' '*(5-len(percent)), end="\r")
    energy_error /= len(test)
    forces_error /= len(test)
    print('\nTest MAE:')
    print('    energy: {:10.3f} kcal/mol'.format(energy_error))
    print('    forces: {:10.3f} kcal/mol/\u212B'.format(forces_error))
    Y -= X.min()
    X -= X.min()
    with open(os.path.join(forcetut, 'energy.txt'),'w') as f:
        for x, y in zip(X,Y):
            f.write(str(x.item())+' '+str(y.item())+'\n')

if False:
    best_model = torch.load(os.path.join(forcetut, 'best_model'))
    # Generate a directory for the ASE computations
    ase_dir = os.path.join(forcetut, 'ase_calcs')
    if not os.path.exists(ase_dir):
        os.mkdir(ase_dir)
    # Write a sample molecule
    molecule_path = os.path.join( ase_dir, 'ethanol.xyz')
    io.write(molecule_path, atoms, format='xyz')
    ethanol_ase = spk.interfaces.AseInterface(
        molecule_path,
        best_model,
        ase_dir,
        device,
        energy=MD17.energy,
        forces=MD17.forces,
        energy_units='kcal/mol',
        forces_units='kcal/mol/A'
    )
    ethanol_ase.optimize(fmax=1e-4)
