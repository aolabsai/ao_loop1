# Welcome to the tutorial to setup AO agent and use it through api call. 

# Third-party libraries
import requests
import numpy as np




                                    # ----------- Helper Functions -----------#

## redefine binary conversion function according to need.

def convert_to_binary(input_to_agent_scaled, scale=10):
    input_to_agent = []
    for i in input_to_agent_scaled:
        likelihood = np.zeros(scale, dtype=int)
        likelihood[0:i] = 1
        input_to_agent += likelihood.tolist()
    return input_to_agent





                                        # ----------- Initialize Kennel -----------#
    #--- We are defining an architecture through api call and we can call this architecture (kennel) to initialize an agent ---#
url = "https://api.aolabs.ai/v0dev/kennel"

payload = { 
    "kennel_id": "tutorial_kennel",  ## --> give e unique name to kennel, multiple agents can be build on one kennel name. 
    "description": "write your discription of kennel",
    "permissions": "private",
    "email": "kushagra@aolabs.ai",  ## Mendatory for assigning developer id. 
    "arch": {
        "arch_i": "[10, 10, 10]",
        "arch_z": "[5]",
        "connector_function": "full_conn"
    } 
    }
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)



                                        # ------------ Agent Training (Individual Input) 👇 ------- #
                    # --- This below code shows how to initialize the agent and train with a individual input. --- #                    

url = "https://api.aolabs.ai/v0dev/kennel/agent"

payload = {
    "kennel_id": "tutorial_kennel",
    "agent_id": "tutorial_agent",
    "email": "kushagra@aolabs.ai",
    "INPUT": "110100011111010001111101000111",
    "LABEL": "11100"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)



                                            # ----------- Pre-train with Baseline Examples -----------#

# Optional - Use this to train the agent on a baseline (if the agent has no prior training, it would output random;
# if it only has 1 label/training event, it can only ever output that until trained on more examples)

training_data = [
    ([2, 1, 3], [5]),
    ([1, 1, 9], [3]),
    ([3, 3, 7], [2]),
    ([0, 3, 6], [2]),
    ([2, 2, 4], [5]),

]
url = "https://api.aolabs.ai/v0dev/kennel/agent"
for inp, label in training_data:
    inp = convert_to_binary(inp, scale=10)
    label = convert_to_binary(label, scale=5)
    inp = ''.join(str(bit) for bit in inp)
    label = ''.join(str(bit) for bit in label)
    print
  
    ## -- api call for agent.next_state 
    payload = {
        "control": { "US": True },
        "kennel_id": "tutorial_kennel",
        "agent_id": "tutorial_agent",
        "email": "kushagra@aolabs.ai",
        "INPUT": inp,
        "LABEL": label
    }
    headers = {
        "accept": "application/json",
        "content-type": "application/json",
        "X-API-KEY": "buildBottomUpRealAGI"
    }

    response = requests.post(url, json=payload, headers=headers)

print("Training Response:", response.text)
                                         


                                            # ----------- Inference on Content ----------- #

# Assume extracted_features = [4, 8, 3] and converted to binary
inp = convert_to_binary([4, 8, 3], scale=10)
binary_inp = ''.join(str(bit) for bit in inp)

# Inference API call (no label)
url = "https://api.aolabs.ai/v0dev/kennel/agent"

payload = {
    "control": { "US": True },
    "kennel_id": "tutorial_kennel",
    "agent_id": "tutorial_agent",
    "email": "kushagra@aolabs.ai",
    "INPUT": binary_inp
}

headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)
print("Inference Response:", response.text)
 


                                            # ----------- Feedback Loop -----------#



# Convert feedback label to binary string
label = convert_to_binary([4], scale=5)  # feedback label 
binary_label = ''.join(str(bit) for bit in label)

# Feedback API call
payload = {
    "control": { "US": True },
    "kennel_id": "tutorial_kennel",
    "agent_id": "tutorial_agent",
    "email": "kushagra@aolabs.ai",
    "INPUT": binary_inp,
    "LABEL": binary_label
}

response = requests.post(url, json=payload, headers=headers)
print("Feedback Response:", response.text)




                                            # ----------- Additional Test Input -----------#


test_input = convert_to_binary([0, 10, 10], scale=10)
binary_test_input = ''.join(str(bit) for bit in test_input)

# Test Inference API call
payload = {
    "control": { "US": True },
    "kennel_id": "tutorial_kennel",
    "agent_id": "tutorial_agent",
    "email": "kushagra@aolabs.ai",
    "INPUT": binary_test_input
}

response = requests.post(url, json=payload, headers=headers)
print("Additional Test Inference:", response.text)
