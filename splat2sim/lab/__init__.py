"""Isaac Lab Task environment for the splat2sim pipeline (``Splat2Sim-FR3-Cubes-v0``).

Import layout (so the pure helpers stay importable without Isaac):

* ``geometry``  numpy/stdlib only: pose -> camera offset, intrinsics, table box, cube spawn sampling
* ``asset``     FR3 + Robotiq 2F-85 USD composition (``pxr`` is imported lazily, only when building)
* ``env_cfg``   Isaac Lab config classes (imports ``isaaclab``; call after ``launch_simulation``/``SimulationApp`` or
                at least with the Lab packages importable)
* ``mdp``       observation / event functions
* ``scripted``  scripted pick-and-lift controller + success metrics

Use ``splat2sim.lab.register()`` (or ``import splat2sim.lab.env_cfg``) to register the gym id.
"""

ENV_ID = "Splat2Sim-FR3-Cubes-v0"


def register() -> None:
    """Register ``Splat2Sim-FR3-Cubes-v0`` with gymnasium (idempotent)."""
    from .env_cfg import register as _register

    _register()
