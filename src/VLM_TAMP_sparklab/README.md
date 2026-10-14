This folder is the bridge for any VLM to control the robot through tool calling using code. The idea is that we predefined bunch of tools that the VLM can use on the robot.

Libary we will be using:
- graspgenx (already setup in /home/sparklab/Documents/cap-x/capx with YAM original gripper, might need to adjust the length for our gripper)
- cuRobo (collison avoidant, motion planning, ...)
- open3d (object tracking)
- Segmentation stuffs 

Perception:
- get_scene_state: scene graph where each node is object label with its tracked point cloud and position
- get_point_cloud(prompt or pointing or both ): I dont remember, but check out sparklab@localhost:/home/sparklab/Documents/cap-x/capx
- Get grasp

Action:
- Pick
- Place
- Move to (position)
- Set gripper (0 - 1) 
- Hand over (move object between arms)
