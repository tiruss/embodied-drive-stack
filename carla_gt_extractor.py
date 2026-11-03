"""
Extract Real 3D Bounding Boxes directly from CARLA Actor Transforms in Ego Frame.
"""
import math
import carla
import numpy as np

def get_actor_3d_box_in_ego(actor: carla.Actor, ego_vehicle: carla.Vehicle) -> tuple[np.ndarray, int]:
    """
    Computes exact relative 3D Bounding Box of an actor in Ego Vehicle Frame:
    Returns:
      box_9d: [cx, cy, cz, w, l, h, yaw_rel, vx, vy]
      class_id: 1 (Car/Vehicle) or 2 (Pedestrian/Walker)
    """
    ego_tf = ego_vehicle.get_transform()
    ego_loc = ego_tf.location
    ego_yaw = math.radians(ego_tf.rotation.yaw)

    actor_tf = actor.get_transform()
    actor_loc = actor_tf.location
    actor_yaw = math.radians(actor_tf.rotation.yaw)

    # Relative position in World Frame
    dx_world = actor_loc.x - ego_loc.x
    dy_world = actor_loc.y - ego_loc.y
    dz_world = actor_loc.z - ego_loc.z

    # Rotate into Ego Vehicle Frame (X=Forward, Y=Right, Z=Up)
    # R_ego = [ [cos, sin], [-sin, cos] ]
    c, s = math.cos(ego_yaw), math.sin(ego_yaw)
    cx_ego = c * dx_world + s * dy_world
    cy_ego = -s * dx_world + c * dy_world
    cz_ego = dz_world

    # Relative yaw angle normalized to [-pi, pi]
    rel_yaw = actor_yaw - ego_yaw
    rel_yaw = (rel_yaw + math.pi) % (2.0 * math.pi) - math.pi

    # Dimensions
    bb = actor.bounding_box
    extent = bb.extent  # half-lengths
    length = extent.x * 2.0
    width = extent.y * 2.0
    height = extent.z * 2.0

    # Velocity in Ego Frame
    vel_world = actor.get_velocity()
    vx_ego = c * vel_world.x + s * vel_world.y
    vy_ego = -s * vel_world.x + c * vel_world.y

    class_id = 2 if "walker" in actor.type_id else 1
    box_9d = np.array([cx_ego, cy_ego, cz_ego, width, length, height, rel_yaw, vx_ego, vy_ego], dtype=np.float32)
    return box_9d, class_id
