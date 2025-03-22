"""*******************************************************************************
* HumARConoid
*
* Advanced Humanoid Locomotion Strategy using Reinforcement Learning
*
*     https://github.com/S-CHOI-S/HumARConoid.git
*
* Advanced Robot Control Lab. (ARC)
* 	  @ Korea Institute of Science and Technology
*
*	  https://sites.google.com/view/kist-arc
*
*******************************************************************************"""

"* Authors: Sol Choi (Jennifer) *"

import os
import re
import xml.etree.ElementTree as ET
import yaml
from collections import defaultdict, deque


def parse_urdf(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    joint_order = []
    urdf_tree = defaultdict(list)
    joint_to_child_link = {}

    for joint in root.findall("joint"):
        joint_name = joint.get("name")
        parent_link = joint.find("parent").get("link")
        child_link = joint.find("child").get("link")

        joint_order.append(joint_name)
        urdf_tree[parent_link].append(joint_name)
        joint_to_child_link[joint_name] = child_link

    return joint_order, urdf_tree, joint_to_child_link


def bfs_joints(tree, joint_to_child_link, start):
    queue = deque([start])
    bfs_order = []

    while queue:
        node = queue.popleft()
        if node in tree:
            upper_case_joints = [j for j in tree[node] if j[0].isupper()]
            lower_case_joints = [j for j in tree[node] if not j[0].isupper()]

            for joint in upper_case_joints + lower_case_joints:
                bfs_order.append(joint)
                child_link = joint_to_child_link[joint]
                queue.append(child_link)

    return bfs_order


def save_to_yaml(data, output_file):
    with open(output_file, "w") as file:
        yaml.dump(data, file, default_flow_style=False, allow_unicode=True)


def main():
    start_link = "kimanoid"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(script_dir, "../exts/humarconoid/humarconoid/robots/KIST_HUMANOID_TORSO/urdf")
    urdf_file_path = os.path.join(file_path, "kist_humanoid3.urdf")
    yaml_file_path = os.path.join(file_path, f"../{start_link}.yaml")

    joint_order, urdf_tree, joint_to_child_link = parse_urdf(urdf_file_path)
    joints_in_bfs_order = bfs_joints(urdf_tree, joint_to_child_link, start_link)

    pattern = re.compile(r"([A-Za-z]+)(\d+)")

    sorted_elements = sorted(joints_in_bfs_order, key=lambda x: (int(pattern.match(x).group(2)),))

    save_to_yaml({"joint_order": joint_order, "sorted_joint": sorted_elements}, yaml_file_path)

    print("joint order:         ", joint_order)
    print("joint from BFS:      ", joints_in_bfs_order)
    print("sorted joint names:  ", sorted_elements)


if __name__ == "__main__":
    # run the main execution
    main()
