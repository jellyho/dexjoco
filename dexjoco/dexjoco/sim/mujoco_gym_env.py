from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Optional

import gymnasium as gym
import mujoco
import numpy as np


@dataclass(frozen=True)
class GymRenderingSpec:
    height: int = 640
    width: int = 640
    camera_id: str | int = -1
    mode: Literal["rgb_array", "human"] = "rgb_array"


class MujocoGymEnv(gym.Env):
    """MujocoEnv with gym interface."""

    # Attribute names holding a MuJoCo body id whose LIVE pose is part of the task's state.
    # Declared per task; the base resolves them, so adding a task is one line.
    #
    # WHY THIS IS NEEDED AT ALL. The recorded `state` already carries object information, but
    # it is the INITIAL pose: `spray_ori_pose`, `plant_ori_pose` and friends are constant for
    # the whole episode -- measured, not assumed, on water_plant demo 10, where all 15 of those
    # dimensions are bit-identical across 309 rows. They exist so `state_restorers` can put a
    # freshly reset scene back to `state[0]`, which is a different job from telling a policy
    # where the object is now.
    #
    # Every env already holds the handles (`self._spray_body_id`, `self._hammer_body_id`, ...),
    # so this reads them rather than introducing anything new.
    LIVE_OBJECT_BODIES: tuple = ()

    def live_object_poses(self) -> dict:
        """{name: [x, y, z, qw, qx, qy, qz]} for each declared body, at the current step.

        A declared attribute is usually one body id, but not always: bimanual_hanoi keeps
        `_disk_body_id` as a {disk name: id} dict, because the task has a stack of them. Both
        shapes are handled, and a dict contributes one entry per key in sorted order so the
        vector's layout is a property of the model rather than of dict insertion order.
        """
        out = {}
        for attr in self.LIVE_OBJECT_BODIES:
            handle = getattr(self, attr, None)
            if handle is None:
                continue
            name = attr.lstrip('_')
            if name.endswith('_body_id'):
                name = name[: -len('_body_id')]
            if isinstance(handle, dict):
                ids = [('%s_%s' % (name, k), v) for k, v in sorted(handle.items())]
            else:
                ids = [(name, handle)]
            for label, body_id in ids:
                out['%s_pose' % label] = np.concatenate(
                    [np.asarray(self._data.xpos[int(body_id)], np.float64),
                     np.asarray(self._data.xquat[int(body_id)], np.float64)]
                )
        return out

    def __init__(
        self,
        xml_path: Path,
        seed: int = 0,
        control_dt: float = 0.02,
        physics_dt: float = 0.002,
        time_limit: float = float("inf"),
        render_spec: GymRenderingSpec = GymRenderingSpec(),
    ):
        self._model = mujoco.MjModel.from_xml_path(xml_path.as_posix())
        self._model.vis.global_.offwidth = render_spec.width
        self._model.vis.global_.offheight = render_spec.height
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = physics_dt
        self._control_dt = control_dt
        self._n_substeps = int(control_dt // physics_dt)
        self._time_limit = time_limit
        self._random = np.random.RandomState(seed)
        self._viewer: Optional[mujoco.Renderer] = None
        self._render_specs = render_spec

    def render(self):
        if self._viewer is None:
            self._viewer = mujoco.Renderer(
                model=self._model,
                height=self._render_specs.height,
                width=self._render_specs.width,
            )
        self._viewer.update_scene(self._data, camera=self._render_specs.camera_id)
        return self._viewer.render()

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()
            self._viewer = None

    def time_limit_exceeded(self) -> bool:
        return self._data.time >= self._time_limit

    # Accessors.

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def data(self) -> mujoco.MjData:
        return self._data

    @property
    def control_dt(self) -> float:
        return self._control_dt

    @property
    def physics_dt(self) -> float:
        return self._model.opt.timestep

    @property
    def random_state(self) -> np.random.RandomState:
        return self._random
