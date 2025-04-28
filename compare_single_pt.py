import numpy as np
import h5py
from io import StringIO
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds
from openff.toolkit import Molecule, ForceField, Topology
from openff.interchange import Interchange
from openff.units import unit
import openmm
from openff.units.openmm import from_openmm, to_openmm
import matplotlib.pyplot as plt

atomic_number_to_symbol = {6: 'C', 1: 'H', 7: 'N', 8: 'O', 15: 'P', 16: 'S'}

datafiles = ["DES370K.hdf5"]

ff_name = "openff-2.0.0.offxml"

openff_interaction_energies = []
openff_ccsdt_energies = []

for datafile in datafiles:
    print(datafile)
    file = h5py.File(datafile, 'r')
    interaction_energies = []
    ccsdt_energies = []
    for set in file.keys():
        try:
            dset = file[set]
            smiles = [str(smile)[2:-1] for smile in dset['smiles']] # strip the 'b and ' off the string
            if not all(all(char == 'C' for char in smile) for smile in smiles): # if not alkanes only then skip
                continue
            print(smiles)
            rdkit_mols = []
            n_mols = dset.attrs['n_molecules']
            for i in range(n_mols):
                xyz_stringio = StringIO()
                xyz_stringio.write(f"{dset['n_atoms'][i]}\n\n")
                atom_offset = 0
                for j in range(i):
                    atom_offset += dset['n_atoms'][j]
                for z, x in zip(dset['atomic_numbers'][atom_offset:atom_offset+dset['n_atoms'][i]], dset['coordinates'][0][atom_offset:atom_offset+dset['n_atoms'][i]]):
                    xyz_stringio.write(f"{atomic_number_to_symbol[z]} {x[0]:f} {x[1]:f} {x[2]:f}\n")
                mol_from_xyz_block = Chem.rdmolfiles.MolFromXYZBlock(xyz_stringio.getvalue())
                if dset['n_atoms'][i] > 1:
                    rdDetermineBonds.DetermineBonds(mol_from_xyz_block, charge=int(dset['mol_charges'][i]))
                mol_from_xyz_block.UpdatePropertyCache()
                rdkit_mols.append(mol_from_xyz_block)
            openff_mols = [Molecule.from_rdkit(mol) for mol in rdkit_mols]
            sage = ForceField(ff_name)
            interchange = Interchange.from_smirnoff(topology=openff_mols, force_field=sage, box=None)
            openmm_sys = interchange.to_openmm()
            openmm_integrator = openmm.VerletIntegrator((1 * unit.femtosecond).m_as("picoseconds"))
            openmm_context = openmm.Context(openmm_sys, openmm_integrator, openmm.Platform.getPlatformByName("Reference"))
            for pos_idx in range(dset['coordinates'].shape[0]):
                positions = dset['coordinates'][pos_idx] / 10.0
                openmm_context.setPositions(positions)
                openmm_state = openmm_context.getState(getEnergy=True)
                total_energy = from_openmm(openmm_state.getPotentialEnergy()).magnitude
                position_offsets = np.zeros((dset['n_atoms'][0], 3))
                for i in range(1, n_mols):
                    new_offset = np.zeros((dset['n_atoms'][i], 3)) + np.array([i * 100, 0, 0])
                    position_offsets = np.vstack((position_offsets, new_offset))
                openmm_context.setPositions(positions + position_offsets)
                openmm_state = openmm_context.getState(getEnergy=True)
                iso_energy = from_openmm(openmm_state.getPotentialEnergy()).magnitude
                interaction_energy = total_energy - iso_energy
                ccsdt_energy = dset['ccsd(t)_cbs.energy'][pos_idx]
                # print(interaction_energy, ccsdt_energy)
                openff_interaction_energies.append(interaction_energy)
                openff_ccsdt_energies.append(ccsdt_energy)
        except Exception as error:
            print(error)
            pass

openff_interaction_energies = np.array(openff_interaction_energies)
openff_ccsdt_energies = np.array(openff_ccsdt_energies)

print(openff_interaction_energies)
print(openff_ccsdt_energies)
