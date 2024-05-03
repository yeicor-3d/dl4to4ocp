import logging
from typing import List

from OCP.TopoDS import TopoDS_Shape
from build123d import Shape, Vector, Location, Box, Part, Align

from dl4to4ocp.mlogging import mlogger


def _solid_grow_bounds(solid: TopoDS_Shape, min: Vector, max: Vector, eps: float = 1e-5) -> TopoDS_Shape:
    """Ensures that the solid is within the specified bounds."""
    solid = Part(solid)
    solid += Location(min) * Box(eps, eps, eps, align=Align.MIN)
    solid += Location(max) * Box(eps, eps, eps, align=Align.MAX)
    return solid.wrapped


def _ensure_shapes_same_cuboid_bb(*solids: TopoDS_Shape) -> List[TopoDS_Shape]:
    """Ensures that all solids have the same size."""
    # Compute the bb of all solids
    bb = None
    for solid in solids:
        if bb is None:
            bb = Shape(solid).bounding_box()
        else:
            bb = bb.add(Shape(solid).bounding_box())

    # Grow each solid to the new bb
    return [_solid_grow_bounds(solid, bb.min, bb.max) for solid in solids]


if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    half_box = Box(0.5, 1, 1)
    mlogger.info(half_box.bounding_box())

    half_box_offset = Location(Vector(2, 0, 0)) * Box(0.5, 1, 1)
    mlogger.info(half_box_offset.bounding_box())

    mlogger.info("Converting to same bb...")
    new_solids = _ensure_shapes_same_cuboid_bb(half_box.wrapped, half_box_offset.wrapped)
    mlogger.info(Shape(new_solids[0]).bounding_box())
    mlogger.info(Shape(new_solids[1]).bounding_box())
