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

import yaml

from humarconoid.robots import HUMARCONOID_EXT_DIR


def load_from_yaml(robot_name):
    file_path = f"{HUMARCONOID_EXT_DIR}/KAPEX/{robot_name}.yaml"
    with open(file_path) as file:
        data = yaml.safe_load(file)

    # [Usage]
    # print("joint_order:", data["joint_order"])
    # print("sorted_joint:", data["sorted_joint"])

    return data


def joint_mapping(original, sorted):
    # Mapping dictionary
    joint_mapping = {}

    # Mapping indices
    for idx, joint in enumerate(original):
        if joint in sorted:
            sorted_index = sorted.index(joint)
            joint_mapping[joint] = {"original_index": idx, "sorted_index": sorted_index}

    # [Usage] Output the mapping
    # print("Joint Index Mapping:")
    # for joint, indices in joint_mapping.items():
    #     print(f"{joint}: joint_order_index={indices['original_index']}, sorted_joint_index={indices['sorted_index']}")

    return joint_mapping
