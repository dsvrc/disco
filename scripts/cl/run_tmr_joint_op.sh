#!/bin/bash
# tmr_joint_neff matching the temporal operator (second moment) as well as the mean, ground-truth replay, on aistpp_genre5, sessions 1..4 (S0 shared): see run_method.sh
#   bash scripts/cl/run_tmr_joint_op.sh [first_session=1] [last_session=4]
exec bash "$(dirname "$0")/run_method.sh" tmr_joint_op "$@"
