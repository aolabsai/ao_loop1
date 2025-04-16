# -*- coding: utf-8 -*-
## // Basic ARC -- Reference Design #3
# 
# The simplest agent we could conceive of for the ARC-AGI benchmark (see arcprize.org).
#
# For interactive visual representation of this Arch:
#    https://miro.com/app/board/uXjVM_kESvI=/?share_link_id=72701488535
#
# Customize and upload this Arch to our API to create Agents: https://docs.aolabs.ai/reference/kennelcreate
#

import ao_pyth as ao

api_key = "my_key"
email = "yours@email.com"


description = "Basic ARC - an agent for the ARC-AGI benchmark"
arch_i = [1, 1, 1,
          1, 1, 1,
          1, 1, 1]
arch_z = [1, 1, 1,
          1, 1, 1,
          1, 1, 1]
arch_c = []           # adding 1 control neuron which we'll define with the instinct control function below
connector_function = "nearest_neighbour_conn"
connector_parameters = [1, 1, 3, 3, False]

arch = ao.Arch(arch_i, arch_z, arch_c, connector_function, connector_parameters, description, api_key=api_key, email=email)