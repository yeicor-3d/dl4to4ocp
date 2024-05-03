from dataclasses import dataclass

from OCP.TopoDS import TopoDS_Compound
from build123d import Vector
from dl4to.solution import Solution as DL4TOSolution

from dl4to4ocp.mlogging import log_timing
from dl4to4ocp.voxels import VoxelsSDF


@dataclass
class Solution(object):
    """Helper class that wraps a dl4to solution and converts it to an OpenCascade shape."""

    dl4to_solution: DL4TOSolution
    """The dl4to solution to be converted."""

    min: Vector
    """The minimum corner of the bounding box of the voxels."""

    max: Vector
    """The maximum corner of the bounding box of the voxels."""

    def to_voxels(self) -> VoxelsSDF:
        """Converts the dl4to solution to a VoxelsSDF object.

        Warning: density distribution is not really a signed distance field."""

        with log_timing("Solution.to_voxels > to numpy"):
            samples = self.dl4to_solution.θ.detach()[0].numpy()

        return VoxelsSDF(samples, self.min, self.max)

    def to_ocp(self) -> TopoDS_Compound:
        """Converts the dl4to solution to an OpenCascade shape, use [to_voxels] for more control."""
        # Warning: density distribution is not really a signed distance field, so MC is used instead of DC
        return self.to_voxels().to_ocp(threshold=0.5, pad=True, impl='mc')
