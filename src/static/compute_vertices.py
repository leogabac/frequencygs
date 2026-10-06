# =============================================================
# God bless whoever reads this code
# Author: leogabac
# =============================================================

import os
import sys
import numpy as np
import pandas as pd
import math
import itertools

from tqdm import tqdm
from math import isclose
from numba import jit, prange, float64, int64, complex128

# own
sys.path.insert(0, "../../../icenumerics/")
sys.path.insert(0, "../auxnumerics/")
sys.path.insert(0, "../")
import icenumerics as ice
import vertices as vrt

# parameters
from parameters import params

ureg = ice.ureg
idx = pd.IndexSlice

if len(sys.argv) != 2:
    print("Usage: python compute_vertices.py <testXX>")
    sys.exit(1)

script_name = sys.argv[0][:-3]
usr_input = sys.argv[1]

# I might sometimes provide a single size so what usr_input is testXX/size/
parts = usr_input.split("/")
if len(parts) > 1:
    has_pre_dir = True
else:
    has_pre_dir = False

DATA_PATH = f"/home/frieren/BIG/reentrancy/{usr_input}/"
SIZES = next(os.walk(DATA_PATH))[1]
REALIZATIONS = 10

# first i want to loop all possible sizes
for strsize in SIZES:
    print(f"N: {strsize}")
    if has_pre_dir:
        params["size"] = int(parts[1])
    else:
        params["size"] = int(strsize)

    # creating the respective folders
    # trj_path = os.path.join(DATA_PATH,strsize,"trj")
    # vrt_path = os.path.join(DATA_PATH,strsize,"vertices")
    # try:
    #     os.mkdir(vrt_path)
    # except:
    #     pass

    # creating the topology
    vrt_lattice = vrt.create_lattice(
        params["lattice_constant"].magnitude, params["size"]
    )
    for i in range(REALIZATIONS + 1):
        trj_file = os.path.join(DATA_PATH, strsize, f"xtrj{i}.csv")
        vrt_file = os.path.join(DATA_PATH, strsize, f"vertices{i}.csv")

        if os.path.isfile(vrt_file):
            print(f"{vrt_file} exists")
            continue

        # Importing files
        try:
            trj_obj = ice.trajectory(trj_file)
            trj_obj.load()
        except:
            continue

        # Doing shit with the vertices
        frames = trj_obj.trj.index.get_level_values("frame").unique().to_list()[::20]

        for frame in tqdm(frames):
            # here the idea is to go frame by frame computing the topological charges
            # and generate the same structure than the vertices module from icenumerics

            # select the current frame, i could have done a group_by('frame') tehee :p
            sel_trj = trj_obj.slice(frame)
            centers, dirs, rels = vrt.trj2numpy(sel_trj)

            # here i make sure the directions are normalized
            dirs = dirs / np.max(dirs)

            # topology shenanigans
            idx_lattice = vrt.indices_lattice(
                vrt_lattice,
                centers,
                params["lattice_constant"].magnitude,
                params["size"],
            )
            q_frame = vrt.get_charge_lattice(idx_lattice, dirs)
            mask = np.where(q_frame == 0, 1, 0)  # put 0 in nonzero, and 1 in zero
            # make all charged vertices have zero dipole
            dipoles = (
                vrt.dipole_lattice(centers, dirs, rels, vrt_lattice, idx_lattice)
                * mask[:, :, np.newaxis]
            )

            # now is time to reshape
            vrt_coord_list = vrt_lattice.reshape(params["size"] ** 2, 3)
            dip_list = dipoles.reshape(params["size"] ** 2, 3)
            q_list = q_frame.reshape(-1)

            # put together
            N = len(q_list)
            data = np.column_stack(
                (
                    [frame] * N,
                    list(range(N)),
                    vrt_coord_list[:, 0],
                    vrt_coord_list[:, 1],
                    [4] * N,
                    q_list,
                    dip_list[:, 0],
                    dip_list[:, 1],
                )
            )

            df = pd.DataFrame(
                data,
                columns=[
                    "frame",
                    "vertex",
                    "x",
                    "y",
                    "coordination",
                    "charge",
                    "dx",
                    "dy",
                ],
            )

            if frame == 0:
                df.to_csv(vrt_file, index=False)
            else:
                df.to_csv(vrt_file, mode="a", index=False, header=False)
