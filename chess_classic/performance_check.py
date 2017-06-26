#!/usr/bin/env python
# encoding: utf-8
# filename: performance_check.py

import pstats, cProfile

import pyximport
pyximport.install()

import chess_util

cProfile.runctx("chess_util.test()", globals(), locals(), "Profile.prof")

s = pstats.Stats("Profile.prof")
s.strip_dirs().sort_stats("time").print_stats()