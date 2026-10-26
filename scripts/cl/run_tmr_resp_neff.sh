#!/bin/bash
# tmr_joint_neff on the music + denoiser hidden-state trajectory, ground-truth replay, on aistpp_genre5, sessions 1..4 (S0 shared): see run_method.sh
#   bash scripts/cl/run_tmr_resp_neff.sh [first_session=1] [last_session=4]
exec bash "$(dirname "$0")/run_method.sh" tmr_resp_neff "$@"
