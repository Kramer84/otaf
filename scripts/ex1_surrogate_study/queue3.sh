#!/bin/bash
P=/home/kramer84/otaf/.venv/bin/python
cd /home/kramer84/otaf/scripts
$P ex1_surrogate_study/lp_timing.py
$P ex1_surrogate_study/gld_check.py F_relu16_credal_200k_s2
$P ex1_surrogate_study/figures.py F_relu16_credal_200k_s2
$P ex1_surrogate_study/slack_metrics.py
