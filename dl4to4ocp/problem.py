import numpy as np
import pyvista as pv

from dl4to4ocp.mlogging import log_timing, mlogger

pv.set_jupyter_backend = lambda *args: mlogger.info('HACK: IGNORING pyvista.jupyter.backend set to', *args)

from dataclasses import dataclass
from typing import List, Tuple

import torch
from OCP.TopoDS import TopoDS_Compound
from build123d import Vector, Sphere, Shape
from dl4to.criteria import Compliance, VolumeConstraint
from dl4to.pde import FDM, PDESolver
from dl4to.problem import Problem
from dl4to4ocp.solution import Solution
from dl4to.topo_solvers import SIMP, TopoSolver
from torch import Tensor, from_numpy

from dl4to4ocp.resize import _ensure_shapes_same_cuboid_bb
from dl4to4ocp.voxels import VoxelsBool, VoxelsForce


@dataclass
class ProblemSetup(object):
    """A helper class to preprocess and convert the inputs from build123d to dl4to."""

    design_space: TopoDS_Compound
    """The area where material presence can be modified. All other voxels will be kept as they are."""

    predefined: TopoDS_Compound
    """The presence values where the material presence cannot be modified (outside the design space)."""

    boundary_conditions: Tuple[TopoDS_Compound, TopoDS_Compound, TopoDS_Compound] = None
    """The area where the displacements are fixed (in X, Y and Z)."""

    forces: List[Tuple[TopoDS_Compound, Vector]] = None
    """The external forces applied to the structure. Each force is a pair of a shape and a vector."""

    def to_dl4to_tensors(self, max_voxels: int, tessellate_tolerance: float = 0.1,
                         tessellate_angular_tolerance: float = 0.1, threshold: float = 0.0) -> (
            Tuple)[Tuple[float, float, float], Tensor, Tensor, Tensor, Vector, Vector]:
        """Converts the input shapes to dl4to tensors by voxelizing them."""
        self.boundary_conditions = self.boundary_conditions or (Sphere(0), Sphere(0), Sphere(0))  # Empty
        self.forces = self.forces or []  # Empty

        # First, we need to resize the shapes to the same bounding box
        with log_timing("to_dl4to_tensors > _ensure_shapes_same_cuboid_bb"):
            ds, p, bcX, bcY, bcZ, *fs = _ensure_shapes_same_cuboid_bb(
                self.design_space, self.predefined, self.boundary_conditions[0], self.boundary_conditions[1],
                self.boundary_conditions[2], *[s for s, _ in self.forces])

        # Now voxelize the inputs that were aligned
        with log_timing("to_dl4to_tensors > all voxels from_ocp"):
            ds_v = VoxelsBool.from_ocp(ds, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, threshold)
            p_v = VoxelsBool.from_ocp(p, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, threshold)
            bc_x_v = VoxelsBool.from_ocp(bcX, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, threshold)
            bc_y_v = VoxelsBool.from_ocp(bcY, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, threshold)
            bc_z_v = VoxelsBool.from_ocp(bcZ, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, threshold)
            fs_v = [VoxelsForce.from_ocp(fs[i], v, max_voxels, tessellate_tolerance, tessellate_angular_tolerance)
                    for i, (_, v) in enumerate(self.forces)]

        # Post-process: boundary conditions into a single array by concatenating them
        with log_timing("to_dl4to_tensors > boundary conditions stack"):
            bc_v_samples = np.stack((bc_x_v.samples, bc_y_v.samples, bc_z_v.samples)).astype(int)

        # Post-process: design-space + predefined voxels into a single array
        with log_timing("to_dl4to_tensors > design space + predefined"):
            p_v_samples = p_v.samples.astype(int)  # False -> 0, True -> 1
            p_v_samples[ds_v.samples] = -1  # -1 -> design space
            p_v_samples = np.expand_dims(p_v_samples, 0)  # Add the batch dimension

        # Post-process: forces into a single array by adding their samples
        with log_timing("to_dl4to_tensors > sum forces"):
            fs_v = VoxelsForce.sum_forces(*fs_v)
            fs_v_samples = np.moveaxis(fs_v.samples, 3, 0)  # Match the dl4to format

        # Convert to tensors and return
        with log_timing("to_dl4to_tensors > to pytorch"):
            bc_v_t = from_numpy(bc_v_samples)
            p_v_t = from_numpy(p_v_samples)
            fs_v_t = from_numpy(fs_v_samples)
            voxel_size = Vector(p_v.spacing).to_tuple()

        bb = Shape(ds).bounding_box()
        return voxel_size, bc_v_t, p_v_t, fs_v_t, bb.min, bb.max

    def to_dl4to_problem(self, max_voxels: int, e: float = 807.489e6 / 10, nu: float = .35,
                         sigma_ys: float = 26.082e6 / 10,
                         pde_solver: PDESolver = FDM(), to_cuda_if_available: bool = False) -> (
            Tuple)[Problem, Vector, Vector]:
        """Read the dl4to docs for more information and/or only use [to_dl4to_tensors] for maximum control."""
        voxel_size, bc_v_t, p_v_t, fs_v_t, min_v, max_v = self.to_dl4to_tensors(max_voxels)

        with log_timing("to_dl4to_problem > create problem"):
            problem = Problem(e, nu, sigma_ys, list(voxel_size), bc_v_t, p_v_t, fs_v_t, pde_solver, 'ocp-problem',
                              'cuda' if to_cuda_if_available and torch.cuda.is_available() else 'cpu')

        return problem, min_v, max_v

    def solve_helper(self, max_voxels: int, simp: TopoSolver = SIMP(
        criterion=Compliance() + VolumeConstraint(max_volume_fraction=.1, threshold_fct='relu'),
        binarizer_steepening_factor=1.02,
        n_iterations=100,
        lr=0.5,
    )) -> Solution:
        """Read the dl4to docs for more information and/or only use [to_dl4to_tensors] for maximum control."""
        problem, min_v, max_v = self.to_dl4to_problem(max_voxels)
        # problem.plot()

        with log_timing("solve_dl4to_problem > solve problem"):
            solution_raw = simp([problem])[0]

        return Solution(solution_raw, min_v, max_v)
