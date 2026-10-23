#!/bin/bash
# random replay + STAR regularisation (arXiv:2503.01595) on aistpp_genre5, sessions 1..4 (S0 shared): see run_method.sh
#   bash scripts/cl/run_random_star.sh [first_session=1] [last_session=4]
exec bash "$(dirname "$0")/run_method.sh" random_star "$@"
