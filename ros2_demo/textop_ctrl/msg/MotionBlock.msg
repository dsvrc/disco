# MotionBlock message definition
# Contains motion data with timestamp and multiple multi-dimensional arrays

# Index for this motion block
int32 index

# Timestamp for this motion block
builtin_interfaces/Time timestamp

# Multiple multi-dimensional float arrays for different motion terms
# Each array represents a different aspect of motion data
std_msgs/Float32MultiArray joint_positions  # [T,Nq]
std_msgs/Float32MultiArray joint_velocities # [T,Nq]
std_msgs/Float32MultiArray anchor_body_ori  # [T,4]
std_msgs/Float32MultiArray anchor_body_pos  # [T,3]