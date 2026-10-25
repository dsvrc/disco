#!/bin/bash
# the paper's TMR on the joint music + dance trajectory (the only change), ground-truth replay, on aistpp_genre5, sessions 1..4 (S0 shared): see run_method.sh
#   bash scripts/cl/run_tmr_joint.sh [first_session=1] [last_session=4]
exec bash "$(dirname "$0")/run_method.sh" tmr_joint "$@"
