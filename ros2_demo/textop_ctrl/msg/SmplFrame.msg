# One SMPL frame

std_msgs/Header header

# Flattened axis-angle rotations for SMPL joints.
# Layout: [root(3), joint1(3), ..., jointN(3)]
float32[] pose_aa

# Root orientation only (axis-angle, len=3).
float32[3] root_rot

# Root translation xyz in the same coordinate system as the motion representation.
float32[3] root_trans
