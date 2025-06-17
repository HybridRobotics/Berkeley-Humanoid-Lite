import os
import time
import meshcat_shapes
import numpy as np
import qpsolvers
from loop_rate_limiters import RateLimiter
import pink
from pink import solve_ik
from pink.tasks import FrameTask, PostureTask
from pink.utils import custom_configuration_vector
from pink.visualization import start_meshcat_visualizer
import pinocchio as pin


class Solver():
    def __init__(
        self,
        urdf_path: str = "./source/berkeley_humanoid_lite_assets/data/urdf/berkeley_humanoid_lite.urdf"
    ):
        # extract the directory of the urdf file to locate asset files
        urdf_package_path = os.path.dirname(urdf_path)
        self.robot = pin.RobotWrapper.BuildFromURDF(
            urdf_path,
            package_dirs=[urdf_package_path],
            root_joint=pin.JointModelFreeFlyer()
        )

        self.init_base_position = np.array([0.0, 0.0, 0.0])
        self.init_base_orientation = np.array([1.0, 0.0, 0.0, 0.0])

        # meshcat visualizer
        self.visualizer = start_meshcat_visualizer(self.robot)
        self.viewer = self.visualizer.viewer

        # end effectors
        self.end_effectors: list = ["arm_left_elbow_roll", "arm_right_elbow_roll"]
        self.end_effector_ids = [self.robot.model.getFrameId(name) for name in self.end_effectors]

        # create visualization frames for the end effectors
        for name in self.end_effectors:
            meshcat_shapes.frame(self.viewer[name + "_target"], opacity=0.5)
            meshcat_shapes.frame(self.viewer[name + "_vive"], opacity=0.25)
            meshcat_shapes.frame(self.viewer[name], opacity=1.0)

        self.end_effector_tasks = [FrameTask(
            name,
            position_cost=1.0,  # [cost] / [m]
            orientation_cost=0.0,  # [cost] / [rad]
            lm_damping=1.0,  # tuned for this setup
        ) for name in self.end_effectors]

        self.base_tasks = [FrameTask(
            "base",
            position_cost=1000.0,
            orientation_cost=1000.0,
            lm_damping=1.0,
        )]

        if any([(ee_id < 0 or ee_id >= len(self.robot.model.frames)) for ee_id in self.end_effector_ids]):
            raise ValueError("End effector name not found in model")
        else:
            print("End effector ids:", self.end_effector_ids)

        # select QP solver
        self.solver = qpsolvers.available_solvers[0]
        if "quadprog" in qpsolvers.available_solvers:
            self.solver = "quadprog"

        self.last_update_time = time.perf_counter()

        self.q = custom_configuration_vector(
            self.robot,
            # (x, y, z, qw, qx, qy, qz)
            # root_joint=[0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            # Set arm joint positions
            arm_left_shoulder_pitch_joint=0.0,
            arm_left_shoulder_roll_joint=0.0,
            arm_left_shoulder_yaw_joint=0.0,
            arm_left_elbow_pitch_joint=0.0,
            arm_left_elbow_roll_joint=0.0,
        )

    def update(self):
        # pose_data = np.random.rand(len(ee_names), 4, 4)
        # vive_poses = [pin.SE3(quat=pin.Quaternion(R=pose_data[i][0:3, 0:3]), translation=pose_data[i][0:3, -1]) for i in range(len(ee_names))]

        # pin.forwardKinematics(self.robot.model, data, q)
        # pin.framesForwardKinematics(self.robot.model, data, q)
        # pin.updateFramePlacements(self.robot.model, data)

        delta_poses = [pin.SE3.Identity() for _ in range(len(self.end_effectors))]

        delta_poses[0].translation[0] = 0.2 + 0.1 * np.cos(2.0 * time.perf_counter())
        delta_poses[0].translation[1] = 0.2 + 0.1 * np.sin(2.0 * time.perf_counter())
        delta_poses[0].translation[2] = 0.6

        delta_poses[1].translation[0] = 0.2 + 0.1 * np.sin(2.0 * time.perf_counter())
        delta_poses[1].translation[1] = -0.2 - 0.1 * np.cos(2.0 * time.perf_counter())
        delta_poses[1].translation[2] = 0.6

        # # Update task targets
        # end_effector_target = end_effector_task.transform_target_to_world
        # end_effector_target.translation[1] = 0.2 + 0.1 * np.sin(2.0 * t)
        # end_effector_target.translation[2] = 0.2

        # desired_poses = [pin.SE3(delta_poses[i].rotation @ self.robot_last_poses[i].rotation, self.robot_last_poses[i].translation + delta_poses[i].translation)  for i in range(self.num_ee)]
        desired_poses = delta_poses

        configuration = pink.Configuration(self.robot.model, self.robot.data, self.q)

        for i, name in enumerate(self.end_effectors):
            self.viewer[name + "_target"].set_transform(desired_poses[i].np)
            # viewer[name + "_vive"].set_transform(vive_poses[i].np)
            self.viewer[name].set_transform(
                configuration.get_transform_frame_to_world(self.end_effector_tasks[i].frame).np
            )

        for i in range(len(self.end_effector_tasks)):
            self.end_effector_tasks[i].transform_target_to_world = desired_poses[i]
        # self.base_tasks[0].transform_target_to_world = pin.SE3(pin.Quaternion(self.init_base_orientation[[1,2,3,0]]), self.init_base_position)
        self.base_tasks[0].transform_target_to_world = pin.SE3(pin.Quaternion.Identity(), self.init_base_position)

        tasks = self.end_effector_tasks + self.base_tasks

        dt = time.perf_counter() - self.last_update_time
        # self.last_update_time = time.perf_counter()
        # Compute velocity and integrate it into next configuration
        velocity = solve_ik(configuration, tasks, dt, solver=self.solver, safety_break=False)
        configuration.integrate_inplace(velocity, dt)

        print(configuration.q[7:17])

        # Visualize result at fixed FPS
        self.visualizer.display(configuration.q)

        self.q = configuration.q

        return configuration.q


if __name__ == "__main__":
    solver = Solver()
    rate = RateLimiter(frequency=100.0, warn=False)

    while True:
        solver.update()
        rate.sleep()
