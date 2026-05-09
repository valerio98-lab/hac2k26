from robot_descriptions import panda_mj_description
import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg


def get_spec() -> mujoco.MjSpec:
    return mujoco.MjSpec.from_file(panda_mj_description.MJCF_PATH)


# Gains match MuJoCo Menagerie franka_emika_panda/panda.xml.
PANDA_ACTUATOR_12 = BuiltinPositionActuatorCfg(
    target_names_expr=("joint1", "joint2"),
    stiffness=4500.0,
    damping=450.0,
    effort_limit=87.0,
    armature=0.1,
)

PANDA_ACTUATOR_34 = BuiltinPositionActuatorCfg(
    target_names_expr=("joint3", "joint4"),
    stiffness=3500.0,
    damping=350.0,
    effort_limit=87.0,
    armature=0.1,
)

PANDA_ACTUATOR_567 = BuiltinPositionActuatorCfg(
    target_names_expr=("joint5", "joint6", "joint7"),
    stiffness=2000.0,
    damping=200.0,
    effort_limit=12.0,
    armature=0.1,
)

PANDA_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(PANDA_ACTUATOR_12, PANDA_ACTUATOR_34, PANDA_ACTUATOR_567),
    soft_joint_pos_limit_factor=0.9,
)

HOME_KEYFRAME = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.0),
    joint_pos={
        "joint1": 0.0,
        "joint2": 0.0,
        "joint3": 0.0,
        "joint4": -1.57079,
        "joint5": 0.0,
        "joint6": 1.57079,
        "joint7": -0.7853,
    },
    joint_vel={".*": 0.0},
)


def get_panda_robot_cfg() -> EntityCfg:
    return EntityCfg(
        init_state=HOME_KEYFRAME,
        spec_fn=get_spec,
        articulation=PANDA_ARTICULATION,
    )


if __name__ == "__main__":

    from mjlab.entity.entity import Entity

    robot = Entity(get_panda_robot_cfg())
    m = robot.spec.compile()
    print("nq:", m.nq, " nv:", m.nv)
    print("joints:", [m.joint(i).name for i in range(m.njnt)])
