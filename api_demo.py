# Welcome to the tutorial to setup AO agent and use it through api call. 

# Third-party libraries
import requests
import numpy as np


# Importing API keys
from config import ao_apikey, openai_apikey, google_apikey 






                                        # ----------- Initialize Kennel -----------#
    #--- We are defining an architecture through api call and we can call this architecture (kennel) to initialize an agent ---#
url = "https://api.aolabs.ai/v0dev/kennel"

payload = { 
    "kennel_id": "tutorial_kennel",  ## --> give e unique name to kennel, multiple agents can be build on one kennel name. 
    "description": "write your discription of kennel",
    "permissions": "private",
    "email": "kushagra@aolabs.ai",  ## Mendatory for assigning developer id. 
    "arch": {
        "arch_i": "[2, 2, 3]",
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

url = "https://api.aolabs.ai/v0dev/kennel/agent"

payload = {
    "kennel_id": "tutirial_kennel",
    "agent_id": "tutorial_agent",
    "email": "kushagra@aolabs.ai",
    "INPUT": "1101000",
    "LABEL": "10101"
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)