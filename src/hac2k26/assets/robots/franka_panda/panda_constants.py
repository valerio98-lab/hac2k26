from robot_descriptions import panda_mj_description
import mujoco
from mjlab.entity import EntityCfg


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)


HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint4": -1.57079,
        "joint6": 1.57079,
        "joint7": -0.7853,
    },
    joint_vel={".*": 0.0},
)


def get_panda_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        spec_fn=get_spec,
    )


if __name__ == "__main__":
    import mujoco.viewer as viewer

    from mjlab.entity.entity import Entity

    robot = Entity(get_panda_robot_cfg())
    m = robot.spec.compile()
    print("nq:", m.nq, " nv:", m.nv)
    print("joints:", [m.joint(i).name for i in range(m.njnt)])
