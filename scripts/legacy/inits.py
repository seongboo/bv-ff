#
import numpy as np
import sys
import ast

# read control.in file

class Controls:
    #
    def __init__(self,file_name):
        self.val = dict()
        with open(file_name,'r') as fp:
            data = fp.readlines()
        for line in data:
            key,value = (x.strip() for x in line.split("=",1))
            value = value.split("#")[0].strip()
            self.val[key]=value
            if key == "n" or key == "nk":
                self.val[key]=int(value)
            if key == "kappa":
                self.val[key]=float(value)

# class for atomic position
# read POSCAR file
class position_info:
    #
    def __init__(self,filename):
        with open(filename,'r') as fp:
            lines = fp.readlines()

         # 1. lattice vectors : lattice_vectors, reciprocal_lattice_vectors, volume
        line_number = 0
        title = lines[line_number].strip()
        line_number += 1
        length_unit = float(lines[line_number])
        self.lattice_vectors = []
        for i in range(3):
            line_number += 1
            self.lattice_vectors.append([
                float(x) for x in lines[line_number].split()])
        self.lattice_vectors = length_unit * np.array(self.lattice_vectors)
        self.reciprocal_lattice_vectors = np.linalg.inv(self.lattice_vectors)
        lat_vec = self.lattice_vectors
        self.volume = np.dot(np.cross(lat_vec[0],lat_vec[1]),lat_vec[2])
        # 2. species : symbols, n_species, n_atoms_per_species, n_atoms_total
        line_number += 1
        self.symbols = lines[line_number].split()
        self.n_species = len(self.symbols)
        line_number += 1
        self.n_atoms_per_species = [
                int(x) for x in lines[line_number].split()]
        self.n_atoms_total = np.sum(self.n_atoms_per_species)

        # 3. atomic positions : positions
        line_number += 1
        first_character = lines[line_number].split()[0][0]
        if first_character in ('d','D'):
            direct = True
        elif first_character in ('c', 'C'):
            direct = False
        else:
            sys.exit('Wrong input! It should be Direct or Cartesian')

        self.positions = []
        for i in range(self.n_atoms_total):
            line_number += 1
            tmp1 = [x for x in lines[line_number].split()]
            tmp2 = [float(x) for x in tmp1[0:3]]
            self.positions.append(tmp2)
        self.positions = np.array(self.positions)
        if not direct:
            # Convert Cartesian to fractional coordinates
            self.positions = np.dot(self.positions, self.reciprocal_lattice_vectors)

# class for potential parameters
# read param file

potential_names=["coulomb", "repulsive12","lj","bv"]

class parameter_info:
    def __init__(self, filename):
        with open(filename, 'r') as fp:
            lines = fp.readlines()

        self.potential_parameters_species = dict()
        self.potential_names = []
        for line in lines:
            name_line = False
            for potential_name in potential_names:
                if potential_name in line:
                    name = potential_name
                    name_line = True
                    self.potential_parameters_species[name] = []
                    self.potential_names.append(name)
            if not name_line:
                temp = line.split()
                data = [ast.literal_eval(x) for x in temp]
                self.potential_parameters_species[name].append(data)

# Convert parameters from species' to atoms'
def species_to_atoms(pos,par):

    par.potential_parameters_atoms = dict()
    for potential_name in par.potential_names:
        dim = 0
        for number in par.potential_parameters_species[potential_name][0]:
            if type(number) == int:
                dim += 1
        if dim == 1:
            par.potential_parameters_atoms[potential_name] = [0 for x in range(pos.n_atoms_total)]
            for line in par.potential_parameters_species[potential_name]:
                ispec = line[0]
                values = line[1:]
                for i in range(pos.n_atoms_per_species[ispec]):
                    ii = i + sum(pos.n_atoms_per_species[:ispec])
                    par.potential_parameters_atoms[potential_name][ii] = values
        elif dim == 2:
            par.potential_parameters_atoms[potential_name] = \
                [[0 for x in range(pos.n_atoms_total)] for x in range(pos.n_atoms_total)]
            for line in par.potential_parameters_species[potential_name]:
                ispec = line[0]
                jspec = line[1]
                values = line[2:]
                for i in range(pos.n_atoms_per_species[ispec]):
                    ii = i + sum(pos.n_atoms_per_species[:ispec])
                    for j in range(pos.n_atoms_per_species[jspec]):
                        jj = j + sum(pos.n_atoms_per_species[:jspec])
                        par.potential_parameters_atoms[potential_name][ii][jj] = values
                        par.potential_parameters_atoms[potential_name][jj][ii] = values
        else:
            sys.exit("dim should be 1 or 2 for pair potentials")















