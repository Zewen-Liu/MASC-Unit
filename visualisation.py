#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 27 14:08:56 2022

@author: mbaxszlh
"""

from model import *
from model_chapter3 import *

# model = torch.load('models/MASC_p2a16k9l2_fold0/model')
# model = torch.load('models/MASC_p2a16k9l2_fold1/model')
# model = torch.load('models/MASC_p2a16k9l2_fold2/model')
# model = torch.load('models/MASC_p2a16k9l2_fold3/model')
model = torch.load('models/MASC_p2a16k9l2_fold4/model')
GConv_visual(model)
convW_visual(model)
correlation_visual(model)
