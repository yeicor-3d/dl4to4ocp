import os
from typing import Tuple

from build123d import export_stl, Part, BuildSketch, Locations, Location, CM, Circle, Mode, BuildPart, add, Axis, \
    extrude, make_hull, Face, Vector

try:
    from ocp_vscode import show_all
except ImportError:
    show_all = lambda *args, **kwargs: print("WARNING: ocp_vscode not available, won't show the shapes")

from dl4to4ocp import ProblemSetup

import logging


def problem_cad_design(max_vox: int) -> Tuple[Part, Part, Part, Part]:
    """This is the build123d-based design of the problem. Check out the build123d documentation for more details."""

    # Screw holes
    with BuildSketch() as sk_circles:
        with Locations(Location((0, 5 * CM)), Location((0, 0)), Location((5 * CM, 0))):
            Circle(1 * CM)
            Circle(0.5 * CM, mode=Mode.SUBTRACT)

    # Use only 2 voxel in Z for 2D problem (dl4to requirement)
    extrude_height = sk_circles.sketch.bounding_box().size.X * 2 / max_vox

    # Dirichlet boundary conditions: two left holes are fixed in place
    with BuildPart() as _fixed_in_place:
        add(sk_circles.faces().group_by(Axis.X)[0])
        extrude(amount=extrude_height)
    _fixed_in_place = _fixed_in_place.part
    _fixed_in_place.color = (0.5, 1.0, 0.5)

    # External forces are applied to the right hole
    with BuildPart() as _external_forces:
        add(sk_circles.faces().group_by(Axis.X)[-1])
        extrude(amount=extrude_height)
    _external_forces = _external_forces.part
    _external_forces.color = (1.0, 0.5, 0.5)

    # Design space: Only the triangle can have material, and not the holes
    with BuildSketch() as sk_design:
        add(sk_circles.faces())
        make_hull()
        for _face in sk_circles.faces():
            add(Face(_face.outer_wire()), mode=Mode.SUBTRACT)
    with BuildPart() as _design_space:
        add(sk_design)
        extrude(amount=extrude_height)
    _design_space = _design_space.part
    _design_space.color = (0.5, 0.5, 0.5)

    return _design_space, _fixed_in_place + _external_forces, _fixed_in_place, _external_forces


if __name__ == '__main__':
    # Define the problem in CAD
    max_voxels = 20
    design_space, predefined, fixed_in_place, external_forces = problem_cad_design(max_voxels)
    show_all()

    # Optional: enable logging to see the progress
    logging.basicConfig(level=logging.DEBUG)

    # Define the dl4to problem
    setup = ProblemSetup(design_space.wrapped, predefined.wrapped,
                         (fixed_in_place.wrapped, fixed_in_place.wrapped, fixed_in_place.wrapped),
                         [(external_forces.wrapped, Vector(0, -1e5, 0))])

    # Solve the dl4to problem
    # NOTE: You can only create the main tensors for the problems and use the dl4to methods directly
    #       for more control over the process. This is just a helper function to get you started.
    solution = setup.solve_helper(max_voxels)

    # Visualize the solution using dl4to utilities
    # solution.dl4to_solution.plot(solve_pde=True)

    # Convert the solution back to a solid CAD model
    sol_raw = Part(solution.to_ocp())
    sol_cut = sol_raw & design_space  # Solution is limited to the design space
    sol_final = sol_cut + predefined  # Solution depends on predefined shapes

    # Show/export the solution (along with all inputs) with the ocp-vscode viewer
    export_dir = os.path.join(os.path.dirname(__file__), 'export')
    os.makedirs(export_dir, exist_ok=True)
    export_stl(sol_final, os.path.join(export_dir, "solution.stl"))
    show_all()
