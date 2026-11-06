#!/bin/bash
# remaining candidates of the screening, one seed
P=/home/kramer84/otaf/.venv/bin/python
cd /home/kramer84/otaf/scripts
for st in B C D E; do $P ex1_surrogate_study/run.py --stage $st; done
