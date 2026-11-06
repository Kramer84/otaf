#!/bin/bash
P=/home/kramer84/otaf/.venv/bin/python
cd /home/kramer84/otaf/scripts
$P ex1_surrogate_study/run.py --stage F5 --seeds 0 1 2 3 4 --n-surf 5000000
$P ex1_surrogate_study/run.py --stage F3 --seeds 0 1 2 --n-surf 5000000
$P ex1_surrogate_study/run.py --stage G --seeds 0 1 2 3 4
