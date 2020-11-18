'''
Before run this script, please run the following.
export PYTHONPATH=$PYTHONPATH:your/path/to/carla/PythonAPI/carla/dist/carla-0.9.9-py3.7-linux-x86_64.egg
'''

from __future__ import print_function

import glob
import os
import sys
import time

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla

import numpy as np
import matplotlib.pyplot as plt


def left_handed2right_handed(carla_transform):
    # return carla_transform
    x = carla_transform.location.x
    y = -carla_transform.location.y
    z = carla_transform.location.z
    roll = carla_transform.rotation.roll
    pitch = -carla_transform.rotation.pitch
    yaw = -carla_transform.rotation.yaw
    return carla.Transform(
        carla.Location(x=x, y=y, z=z), carla.Rotation(roll=roll, pitch=pitch, yaw=yaw)
    )


class LaneGeometry(object):
    def __init__(self):
        self.center_line = []
        self.left_boundary = []
        self.right_boundary = []


class LaneTopology(object):
    def __init__(self):
        self.predecessors = []
        self.successors = []
        self.left_lane = None
        self.right_lane = None


def in_same_lane(carla_wp1, carla_wp2):
    return (
        carla_wp1.road_id == carla_wp2.road_id
        and carla_wp1.section_id == carla_wp2.section_id
        and carla_wp1.lane_id == carla_wp2.lane_id
    )


def encode_id(road_id, section_id, lane_id):
    return "{}.{}.{}".format(road_id, section_id, lane_id)


def encode_id_from_waypoint(carla_waypoint):
    return encode_id(
        carla_waypoint.road_id, carla_waypoint.section_id, carla_waypoint.lane_id
    )


def decode_id(absolute_id):
    ids = absolute_id.split(".")
    road_id = int(ids[0])
    section_id = int(ids[1])
    lane_id = int(ids[2])
    return road_id, section_id, lane_id


def previous_until_lane_start(carla_waypoint, predecessor, distance=1.0):
    prev_waypoints = []
    seed = predecessor
    count = 0
    while True:
        wp_list = seed.next(distance)
        next_waypoints = filter(lambda wp: in_same_lane(wp, carla_waypoint), wp_list)

        ended = False
        for wp in next_waypoints:
            if wp.s < carla_waypoint.s:
                prev_waypoints.append(wp)
            else:
                ended = True
                break
        if ended:
            break

        if len(next_waypoints) > 0:
            seed = next_waypoints[-1]
        else:
            seed = wp_list[-1]

        count += 1

        if count > 200:
            break

    return prev_waypoints


def next_until_lane_end(carla_waypoint, distance=1.0):
    waypoints = []
    seed = carla_waypoint
    while True:
        wp_list = seed.next(distance)
        next_waypoints = filter(lambda wp: in_same_lane(wp, carla_waypoint), wp_list,)
        if len(next_waypoints) == 0:
            break
        else:
            waypoints += next_waypoints
            seed = waypoints[-1]

    return waypoints


def calc_center_line(carla_waypoint):
    transform = left_handed2right_handed(carla_waypoint.transform)
    return transform


def calc_left_boundary(carla_waypoint):
    transform = left_handed2right_handed(carla_waypoint.transform)
    yaw = transform.rotation.yaw
    x0 = transform.location.x
    y0 = transform.location.y
    z0 = transform.location.z
    x = x0 - 0.5 * carla_waypoint.lane_width * np.sin(np.deg2rad(yaw))
    y = y0 + 0.5 * carla_waypoint.lane_width * np.cos(np.deg2rad(yaw))
    return carla.Transform(carla.Location(x=x, y=y, z=z0), transform.rotation)


def calc_right_boundary(carla_waypoint):
    transform = left_handed2right_handed(carla_waypoint.transform)
    yaw = transform.rotation.yaw
    x0 = transform.location.x
    y0 = transform.location.y
    z0 = transform.location.z
    x = x0 + 0.5 * carla_waypoint.lane_width * np.sin(np.deg2rad(yaw))
    y = y0 - 0.5 * carla_waypoint.lane_width * np.cos(np.deg2rad(yaw))
    return carla.Transform(carla.Location(x=x, y=y, z=z0), transform.rotation)


def pre_and_successors_waypoints(carla_waypoint, topologies):
    predecessors = []
    successors = []
    for pair in topologies:
        src, dest = pair
        if (
            src.road_id == carla_waypoint.road_id
            and src.section_id == carla_waypoint.section_id
            and src.lane_id == carla_waypoint.lane_id
        ):
            if not (
                dest.road_id == carla_waypoint.road_id
                and dest.section_id == carla_waypoint.section_id
                and dest.lane_id == carla_waypoint.lane_id
            ):
                successors.append(dest)

        if (
            dest.road_id == carla_waypoint.road_id
            and dest.section_id == carla_waypoint.section_id
            and dest.lane_id == carla_waypoint.lane_id
        ):
            if not (
                src.road_id == carla_waypoint.road_id
                and src.section_id == carla_waypoint.section_id
                and src.lane_id == carla_waypoint.lane_id
            ):
                predecessors.append(src)
    return predecessors, successors


def plot_lane(lane):
    center_xs = []
    center_ys = []
    for p in lane.center_line:
        center_xs.append(p.location.x)
        center_ys.append(p.location.y)

    left_xs = []
    left_ys = []
    for p in lane.left_boundary:
        left_xs.append(p.location.x)
        left_ys.append(p.location.y)

    right_xs = []
    right_ys = []
    for p in lane.right_boundary:
        right_xs.append(p.location.x)
        right_ys.append(p.location.y)

    plt.plot(center_xs, center_ys, "y-.")
    plt.plot(left_xs, left_ys, "b")
    plt.plot(right_xs, right_ys, "g")
    # start
    plt.plot([left_xs[0], center_xs[0]], [left_ys[0], center_ys[0]], "r--")
    plt.plot([right_xs[0], center_xs[0]], [right_ys[0], center_ys[0]], "r--")
    # end
    plt.plot([left_xs[-1], center_xs[-1]], [left_ys[-1], center_ys[-1]], "r")
    plt.plot([right_xs[-1], center_xs[-1]], [right_ys[-1], center_ys[-1]], "r")


def print_waypoint(wp):
    wp_info = "Road: {}, Section: {}, Lane: {}, S: {}, Width: {}, x:{}, y:{}".format(
        wp.road_id,
        wp.section_id,
        wp.lane_id,
        wp.s,
        wp.lane_width,
        wp.transform.location.x,
        wp.transform.location.y,
    )
    print(wp_info)


def waypoints_distance(wp1, wp2):
    distance = np.power(
        wp1.transform.location.x - wp2.transform.location.x, 2
    ) + np.power(wp1.transform.location.y - wp2.transform.location.y, 2)
    return np.sqrt(distance)


class Lane(object):
    def __init__(self):
        self.geometry = LaneGeometry()
        self.topology = LaneTopology()
        self.lane_id = None
        self.section_id = None
        self.road_id = None
        self._id = None
        self._lane_waypoints = []

    @property
    def id(self):
        return encode_id(self.road_id, self.section_id, self.lane_id)

    @property
    def predecessors(self):
        return self.topology.predecessors

    @property
    def successors(self):
        return self.topology.successors

    @property
    def left_lane(self):
        return self.topology.left_lane

    @property
    def right_lane(self):
        return self.topology.right_lane

    @property
    def center_line(self):
        return self.geometry.center_line

    @center_line.setter
    def center_line(self, center_line):
        self.geometry.center_line = center_line

    @property
    def left_boundary(self):
        return self.geometry.left_boundary

    @left_boundary.setter
    def left_boundary(self, left_boundary):
        self.geometry.left_boundary = left_boundary

    @property
    def right_boundary(self):
        return self.geometry.right_boundary

    @right_boundary.setter
    def right_boundary(self, right_boundary):
        self.geometry.right_boundary = right_boundary

    @property
    def waypoints(self):
        return self._lane_waypoints

    @waypoints.setter
    def waypoints(self, waypoints):
        self._lane_waypoints = waypoints

    def __repr__(self):
        return "Lane({})".format(self.id)

    @staticmethod
    def from_carla_waypoints(waypoints, topologies):
        assert len(waypoints) > 0
        lane = Lane()
        lane_waypoint = waypoints[0]
        lane.road_id = lane_waypoint.road_id
        lane.section_id = lane_waypoint.section_id
        lane.lane_id = lane_waypoint.lane_id
        lane.waypoints = sorted(waypoints, key=lambda wp: wp.s)
        # geometry
        lane.geometry.left_boundary = map(calc_left_boundary, lane.waypoints)
        lane.geometry.right_boundary = map(calc_right_boundary, lane.waypoints)
        lane.geometry.center_line = map(calc_center_line, lane.waypoints)

        # topology
        predecessors, successors = pre_and_successors_waypoints(
            lane_waypoint, topologies
        )
        lane.topology.predecessors = map(
            lambda wp: encode_id_from_waypoint(wp), predecessors,
        )
        lane.topology.successors = map(
            lambda wp: encode_id_from_waypoint(wp), successors,
        )
        left_lane_waypoint = lane_waypoint.get_left_lane()
        if left_lane_waypoint is None:
            lane.topology.left_lane = None
        else:
            lane.topology.left_lane = encode_id_from_waypoint(left_lane_waypoint)

        right_lane_waypoint = lane_waypoint.get_right_lane()
        if right_lane_waypoint is None:
            lane.topology.right_lane = None
        else:
            lane.topology.right_lane = encode_id_from_waypoint(right_lane_waypoint)

        return lane


class OpenDriveManager(object):
    def __init__(self, carla_map, sample_distance=0.2, right_handed=True):
        self.carla_map = carla_map
        self.sample_distance = sample_distance
        self.right_handed = right_handed
        self.topologies = []
        self.lane_pool = {}
        self.__create_lane_pool()

    @property
    def map_name(self):
        return self.carla_map.name

    @property
    def lanes(self):
        return self.lane_pool.values()

    def get_lanes_in_area(self, location, radius):
        def within_area(lane, src_location, radius):
            for p in lane.center_line:
                distance = np.sqrt(
                    np.power(p.location.x - src_location.x, 2)
                    + np.power(p.location.y - src_location.y, 2)
                )
                if distance < radius:
                    return True
            return False

        lanes_in_area = []
        for lane in self.lanes:
            if within_area(lane, location, radius):
                lanes_in_area.append(lane)

        return lanes_in_area

    @staticmethod
    def from_xodr(xodr_file, sample_distance=0.25):
        xodr_string = ""
        with open(xodr_file, "r") as f:
            xodr_string = f.read()
        map_name = xodr_file.split(".")[0]
        carla_map = carla.Map(map_name, xodr_string)
        return OpenDriveManager(carla_map, sample_distance)

    def __create_lane_pool(self):
        lane_waypoints = self.__create_lane_waypoints()
        self.topologies = self.carla_map.get_topology()
        for waypoints in lane_waypoints.values():
            lane = Lane.from_carla_waypoints(waypoints, self.topologies)
            self.lane_pool[lane.id] = lane

    def __create_lane_waypoints(self):
        lane_waypoints = {}
        waypoints = self.carla_map.generate_waypoints(self.sample_distance)
        for wp in waypoints:
            absolute_id = encode_id_from_waypoint(wp)
            if absolute_id not in lane_waypoints:
                lane_waypoints[absolute_id] = []
            lane_waypoints[absolute_id].append(wp)
        return lane_waypoints

    def get_lane_by_id(self, absolute_id):
        if absolute_id in self.lane_pool:
            return self.lane_pool[absolute_id]
        return None

    def get_lane_by_location(self, carla_location):
        waypoint = self.carla_map.get_waypoint(
            carla_location, project_to_road=True, lane_type=carla.LaneType.Any
        )
        if waypoint is None:
            return None
        absolute_id = encode_id_from_waypoint(waypoint)
        if absolute_id in self.lane_pool:
            return self.lane_pool[absolute_id]
        return None

    def get_waypoint(self, carla_location):
        return self.carla_map.get_waypoint(
            carla_location, project_to_road=True, lane_type=carla.LaneType.Any
        )


def plot_map(open_drive_manager):
    for lane in open_drive_manager.lanes:
        print("drawing {}".format(lane))
        plot_lane(lane)


def load_ego_traces(traces_file):
    xs = []
    ys = []
    with open(traces_file, "r") as f:
        for line in f:
            values = line.strip().split(" ")
            # if len(values) != 8:
            #     continue
            frame = int(values[0])
            stamp = float(values[1]) * 1e-6
            x = float(values[2])
            y = float(values[3])
            z = float(values[4])
            roll = float(values[5])
            pitch = float(values[6])
            yaw = float(values[7])

            xs.append(x)
            ys.append(y)
    return xs, ys


if __name__ == "__main__":
    xodr_file = "sample.xodr"
    open_drive_manager = OpenDriveManager.from_xodr(xodr_file, sample_distance=1.0)
    plot_map(open_drive_manager)
    plt.axis("equal")
    plt.show()
