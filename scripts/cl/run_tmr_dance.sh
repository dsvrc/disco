#!/bin/bash
# TMR for dance (motion-aware temporal modes, one clip per dance, distillation replay) on aistpp_genre5, sessions 1..4 (S0 shared): see run_method.sh
#   bash scripts/cl/run_tmr_dance.sh [first_session=1] [last_session=4]
exec bash "$(dirname "$0")/run_method.sh" tmr_dance "$@"
