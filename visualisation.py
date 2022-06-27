#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:56 2022

@author: mbaxszlh
"""

from model import *
from model_chapter3 import *

model = torch.load('models/Chapter3_p2a16k9l2_fold4/model')
GConv_visual(model)
convW_visual(model)
correlation_visual(model)
