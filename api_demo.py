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
    "arch_URL": "https://raw.githubusercontent.com/aolabsai/ao_loop1/refs/heads/main/demo_arch.py",   ##--> you need to define the architecture like this on github
    "description": "write your discription of kennel",
    "permissions": "private",
    "email": "kushagra@aolabs.ai"  ## Mendatory for assigning developer id. 
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

INPUT_AO_api = "1101000"
LABEL = "10101"  
payload = {
    "kennel_id": "tutirial_kennel",  # use kennel_name entered above
    "agent_id": "tutorial_agent",   # enter unique user IDs here, to call a unique agent for each ID.
    "INPUT": INPUT_AO_api,  # pass through the input from embedding_bucketing.auto_sort, adding any other inputs
    "LABEL": LABEL,
    "email": "kushagra@aolabs.ai", 
    "control": {
        "US": True,
        "states": 5,
    }
}
headers = {
    "accept": "application/json",
    "content-type": "application/json",
    "X-API-KEY": "buildBottomUpRealAGI"
}

response = requests.post(url, json=payload, headers=headers)

print(response.text)