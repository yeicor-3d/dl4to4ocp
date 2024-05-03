import logging
import os.path
from dataclasses import dataclass
from typing import Literal, Annotated, TypeVar, Generic, Tuple

import numpy as np
import numpy.typing as npt
import pysdf
import sdftoolbox
from OCP.TopoDS import TopoDS_Solid, TopoDS_Compound
from build123d import Solid, Face, Wire, Edge, Box, Vector, Location, Sphere, Compound, export_stl, Shell
from skimage.measure import marching_cubes

from dl4to4ocp.mlogging import log_timing, mlogger

T = TypeVar("T")


@dataclass
class Voxels(Generic[T]):
    """A 3D voxels-like object, that can hold data of any length."""

    samples: Annotated[npt.NDArray[T], Literal["X", "Y", "Z", "W"]]
    """An array of X by Y by Z samples on the corners of each voxel, each containing W values."""

    min: Vector
    """The minimum corner of the bounding box of the voxels."""

    max: Vector
    """The maximum corner of the bounding box of the voxels."""

    @property
    def num_samples(self) -> np.ndarray:
        """The number of samples in the X, Y and Z directions."""
        return np.array(self.samples.shape)

    @property
    def num_voxels(self) -> np.ndarray:
        """The number of voxels in the X, Y and Z directions."""
        return self.num_samples - 1

    @property
    def center_samples(self):
        """The samples at the center of each voxel, using linear interpolation."""
        return (self.samples[:-1, :-1, :-1] + self.samples[1:, :-1, :-1] + self.samples[:-1, 1:, :-1] +
                self.samples[:-1, :-1, 1:] + self.samples[1:, 1:, :-1] + self.samples[1:, :-1, 1:] +
                self.samples[:-1, 1:, 1:] + self.samples[1:, 1:, 1:]) / 8

    @property
    def spacing(self) -> np.array:
        """The spacing between each voxel in the X, Y and Z directions. This is the same as the voxel size."""
        return np.array((self.max - self.min).to_tuple()) / self.num_voxels


@dataclass
class VoxelsSDF(Voxels[np.float32]):
    """A 3D voxels-like object, that holds the signed distance to a surface for each voxel. The outside is negative."""

    @staticmethod
    def from_ocp(solid: TopoDS_Solid, max_voxels: int, tessellate_tolerance: float = 0.1,
                 tessellate_angular_tolerance: float = 0.1, pad_voxels: float = -1e-2) -> "VoxelsSDF":
        """Converts a CAD solid to signed distance field (SDF) voxels."""
        assert max_voxels >= 1

        with log_timing("from_ocp > bounding box and voxel count"):
            # Grab the exact bounding box of the solid
            solid_bd = Solid(solid)
            bounding_box = solid_bd.bounding_box(None)

            # Get the proper SDF resolution for (almost) cubic voxels
            voxels_multiplier = np.array(bounding_box.size.to_tuple()) / np.max(bounding_box.size.to_tuple())

            # Ensure the border is fully outside the solid by 1 voxel on each edge
            # NOTE: This avoids squished corners when rebuilding the solid with dual contouring
            bounding_box.min -= Vector(
                np.array(bounding_box.size.to_tuple()) * pad_voxels / max_voxels * voxels_multiplier)
            bounding_box.max += Vector(
                np.array(bounding_box.size.to_tuple()) * pad_voxels / max_voxels * voxels_multiplier)

            # Compute the number of samples (voxels + 1) in each direction
            num_voxels = max_voxels * voxels_multiplier
            # dl4to requires at least 2 voxels (3 samples) in each direction
            num_voxels = np.clip(np.round(num_voxels).astype(int), 2, None)
            num_samples = num_voxels + 1  # We sample all corners of the voxels

        with log_timing("from_ocp > tessellate"):
            # Tessellate a triangle mesh to memory
            vertices, faces = solid_bd.tessellate(tessellate_tolerance, tessellate_angular_tolerance)
            vertices_np = np.array([[v for v in vs] for vs in vertices], dtype=np.float32)
            faces_np = np.array([[f for f in fs] for fs in faces], dtype=np.uint32)

        with log_timing("from_ocp > SDF sampling"):
            bb_min = bounding_box.min.to_tuple()
            bb_max = bounding_box.max.to_tuple()
            x = np.linspace(bb_min[0], bb_max[0], num_samples[0])
            y = np.linspace(bb_min[1], bb_max[1], num_samples[1])
            z = np.linspace(bb_min[2], bb_max[2], num_samples[2])
            # TODO: Avoid these loops with a nice numpy trick
            to_sample = np.array([[xv, yv, zv] for xv in x for yv in y for zv in z])
            # WARNING: pysdf is much faster but less accurate than trimesh (not really needed for dl4to)
            # sampled = mesh.nearest.signed_distance(to_sample)
            sampled = pysdf.SDF(vertices_np, faces_np).calc(to_sample)
            # TODO: Avoid these loops with a nice numpy trick
            voxels_sdf = np.zeros(num_samples, dtype=np.float32)
            for i, xv in enumerate(x):
                for j, yv in enumerate(y):
                    for k, zv in enumerate(z):
                        voxels_sdf[i, j, k] = sampled[i * num_samples[1] * num_samples[2] + j * num_samples[2] + k]
            # print("Voxels SDF", voxels_sdf)

        # Return the SDF voxels
        return VoxelsSDF(voxels_sdf, bounding_box.min, bounding_box.max)

    def pad(self, padding: int) -> "VoxelsSDF":
        """Pads the SDF by adding a border of further away values (using spacing)."""
        new_voxels = np.pad(self.samples, padding, mode='edge')
        to_subtract = self.spacing  # Each new edge is spacing further away
        for i in range(padding):
            new_voxels[i, :, :] -= to_subtract[0] * (padding - i)
            new_voxels[-i - 1, :, :] -= to_subtract[0] * (padding - i)
            new_voxels[:, i, :] -= to_subtract[1] * (padding - i)
            new_voxels[:, -i - 1, :] -= to_subtract[1] * (padding - i)
            new_voxels[:, :, i] -= to_subtract[2] * (padding - i)
            new_voxels[:, :, -i - 1] -= to_subtract[2] * (padding - i)
        return VoxelsSDF(new_voxels, self.min - Vector(padding * self.spacing),
                         self.max + Vector(padding * self.spacing))

    def to_trimesh(self, pad: bool = False, threshold: float = 0, impl: Literal['mc', 'dc'] = 'dc') -> tuple[
        np.ndarray, np.ndarray]:
        if pad:
            # Make sure we generate a watertight solid by adding a small padding
            return self.pad(1).to_trimesh(pad=False, threshold=threshold, impl=impl)

        with log_timing(f"to_ocp > isosurface reconstruction with {impl}"):
            # Run an isosurface reconstruction algorithm to convert the SDF to a solid
            if impl == 'mc':
                vertices, faces = self._to_trimesh_mc(threshold)
            else:
                vertices, faces = self._to_trimesh_dc(threshold)

        return vertices, faces

    def _to_trimesh_dc(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        # noinspection PyTypeChecker
        grid = sdftoolbox.grid.Grid(tuple(self.num_samples.tolist()), self.min.to_tuple(), self.max.to_tuple())
        scene = sdftoolbox.sdfs.Discretized(grid, self.samples - threshold)
        vertices, faces = sdftoolbox.dual_isosurface(
            scene, grid, triangulate=True,
            edge_strategy=sdftoolbox.dual_strategies.LinearEdgeStrategy(),
            vertex_strategy=sdftoolbox.dual_strategies.DualContouringVertexStrategy()
        )
        return vertices, faces

    def _to_trimesh_mc(self, threshold: float) -> tuple[np.ndarray, np.ndarray]:
        vertices, faces, _, _ = marching_cubes(self.samples, level=threshold, spacing=self.spacing,
                                               allow_degenerate=False)
        # Displace the vertices to the correct position
        vertices += self.min.to_tuple()
        return vertices, faces

    def to_ocp(self, pad: bool = False, threshold: float = 0, impl: Literal['mc', 'dc'] = 'dc') -> TopoDS_Compound:
        """Converts the SDF voxels to a CAD solid."""
        vertices, faces = self.to_trimesh(pad, threshold, impl)
        mlogger.info("Vertices: %s, Faces: %s", vertices.shape, faces.shape)

        with log_timing("to_ocp > making OCP faces"):
            # Convert the vertices and faces to CAD faces
            all_faces = []
            for face in faces:
                v0 = vertices[face[0]]
                v1 = vertices[face[1]]
                v2 = vertices[face[2]]

                # Ignore degenerate faces
                if np.linalg.norm(np.cross(v1 - v0, v2 - v0)) < 1e-6:
                    continue

                edge1 = Edge.make_line(v0, v1)
                edge2 = Edge.make_line(v1, v2)
                edge3 = Edge.make_line(v2, v0)
                f = Face(Wire([edge1, edge2, edge3]))
                f.wrapped.Reverse()
                all_faces.append(f)

        with log_timing("to_ocp > make compound from faces"):
            # WARNING: Shell(all_faces) sometimes swaps face orientation??
            # return Compound(all_faces).wrapped
            shells = Shell(all_faces).shells()
            solids = [Solid(s) for s in shells]
            return Compound(solids).wrapped

    def plot(self, dist_from: float = 0.0, dist_to: float = None, display: bool = True) -> Tuple[object, object]:
        """Plots the SDF voxels as a 3D image."""
        import matplotlib.pyplot as plt

        fig = plt.figure()

        ax = fig.add_subplot(1, 1, 1, projection='3d')
        m_voxels = self.center_samples
        if dist_from is None:
            dist_from = m_voxels.min()
        if dist_to is None:
            dist_to = m_voxels.max()

        voxels_to_draw = np.bitwise_and(dist_from <= m_voxels, m_voxels <= dist_to)
        # Add nice changing face colors by distance
        colors_src = (m_voxels - dist_from) / (dist_to - dist_from)
        colormap_plasma = plt.get_cmap('plasma')
        colors = colormap_plasma(colors_src)
        colors[..., 3] = 0.5  # Transparency

        ax.voxels(voxels_to_draw, facecolors=colors, edgecolors='k')
        ax.set_title(f"SDF Voxels ({dist_from} to {dist_to})")

        if display:
            plt.show()
        return fig, ax

    def to_force(self, force: Annotated[np.array, Literal["XYZ"]]) -> "VoxelsForce":
        """Converts the SDF voxels to a force voxels object by multiplying the provided force to each voxel."""


@dataclass
class VoxelsBool(Voxels[bool]):
    """A 3D voxels-like object, that holds a single true/false value for each voxel"""

    @staticmethod
    def from_ocp(solid: TopoDS_Solid, max_voxels: int, tessellate_tolerance: float = 0.1,
                 tessellate_angular_tolerance: float = 0.1, threshold: float = 0.0,
                 pad_voxels: float = -1e-2) -> "VoxelsBool":
        """Converts a CAD solid to boolean voxels."""
        voxels_sdf = VoxelsSDF.from_ocp(
            solid, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, pad_voxels)
        return VoxelsBool(voxels_sdf.samples >= threshold, voxels_sdf.min, voxels_sdf.max)


@dataclass
class VoxelsForce(Voxels[np.float32]):
    """A 3D voxels-like object, that holds a 3D force vector for each voxel"""

    @staticmethod
    def from_ocp(solid: TopoDS_Solid, force: Vector, max_voxels: int, tessellate_tolerance: float = 0.1,
                 tessellate_angular_tolerance: float = 0.1, pad_voxels: float = -1e-2) -> "VoxelsForce":
        """Converts a CAD solid to force voxels."""
        voxels_sdf = VoxelsSDF.from_ocp(
            solid, max_voxels, tessellate_tolerance, tessellate_angular_tolerance, pad_voxels)
        new_voxels = np.zeros(voxels_sdf.samples.shape + (3,), dtype=np.float32)
        for i in range(3):
            new_voxels[..., i] = voxels_sdf.samples.clip(0, np.inf) * force.to_tuple()[i]
        return VoxelsForce(new_voxels, voxels_sdf.min, voxels_sdf.max)

    @staticmethod
    def sum_forces(*forces: "VoxelsForce") -> "VoxelsForce":
        """Sums all the forces in the provided force voxels, which must have the same shape."""
        new_voxels = np.zeros(forces[0].samples.shape, dtype=np.float32)
        for f in forces:
            new_voxels += f.samples
        return VoxelsForce(new_voxels, forces[0].min, forces[0].max)


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # test_obj = Box(1, 1, 1)
    test_obj = Location((0, 0, 0), (45, 0, 0)) * (Box(1, 1, 1) - Sphere(0.55))
    export_dir = os.path.join(os.path.dirname(__file__), '..', 'example', 'export')
    export_stl(test_obj, os.path.join(export_dir, "voxels_test.stl"))

    # NOTE: Padding is useful for rebuilding instantly, but not for dl4to
    # NOTE: trimesh was much more accurate for roundtrip conversions, but not needed for dl4to
    voxels = VoxelsSDF.from_ocp(test_obj.wrapped, 16, pad_voxels=1, )

    # voxels.plot()

    test_obj_back = voxels.to_ocp()
    export_stl(Compound(test_obj_back), os.path.join(export_dir, "voxels_test_back.stl"))
