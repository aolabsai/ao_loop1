# -*- coding: utf-8 -*-
"""
// aolabs.ai software >ao_core/Arch.py (C) 2025 Animo Omnis Corporation. All Rights Reserved.
"""


description = "Tutorial architecture for demo"
arch_i = [2, 2, 3]               
arch_z = [5]                       # 10 binary neurons for output-- if the sum of the response >7, positive recommendation
arch_c = []
connector_function = "full_conn"

# To maintain compatibility with our API, do not change the variable name "Arch" or the constructor class "ar.Arch" in the line below
Arch = ao.Arch(arch_i, arch_z, arch_c, connector_function, description)