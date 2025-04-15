# Python standard libraries
import ast

# Third-party libraries
import requests
import numpy as np
from openai import OpenAI

# AO library
import ao_pyth as ao # $ pip install ao_pyth - https://pypi.org/project/ao-pyth/
# import ao_core as ao # private package, to run our code locally, useful for advanced debugging; ao_pyth is enough for most use cases


# Importing API keys
from config import ao_apikey, openai_apikey, google_apikey 





                                    # ----------- Helper Functions -----------#

def llm_call(input_message): #llm call method 
    client = OpenAI(api_key = openai_apikey)
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",  
        messages=[
            {"role": "user", "content": input_message}
        ],
        temperature=0.1
    )
    local_response = response.choices[0].message.content
    return local_response


def get_youtube_data(video_id="dQw4w9WgXcQ"):
    # constructed using grok: https://x.com/i/grok/share/W2HgoXmN638QUOJNKaqtnKiS2
    # refer to that if unsure how to generate your own YT API key
    url = f"https://www.googleapis.com/youtube/v3/videos?part=snippet&id={video_id}&key={google_apikey}"

    response = requests.get(url)
    data = response.json()

    description = data["items"][0]["snippet"]["description"]
    all_data = data
    print(description)
    return description, all_data


def convert_to_binary(input_to_agent_scaled, scale=10):
    input_to_agent = []
    for i in input_to_agent_scaled:
        likelihood = np.zeros(scale, dtype=int)
        likelihood[0:i] = 1
        input_to_agent += likelihood.tolist()
    return input_to_agent





                                    # ----------- Initialize AO Agent -----------#

# Initialize AO agent architecture, here with 30 input neurons and 5 output neurons. 
# Input consists of 3 features, each given on a intensity (or other) scale of 0-10 (10 neurons for each feature):
# Output consists of 5 neurons corresponding to a single scale of 1-5 (or whatever output(s) you want to associate with input).

arch = ao.Arch(arch_i="[10, 10, 10]", arch_z="[5]", api_key=ao_apikey, kennel_id="loop1-demo_01") # --> architecture setup
agent = ao.Agent(arch, uid="agent_01", save_meta=True)  # --> agent creation





                                    # ----------- Pre-train with Baseline Examples -----------#

# Optional - Use this to train the agent on a baseline (if the agent has no prior training, it would output random;
# if it only has 1 label/training event, it can only ever output that until trained on more examples)

training_data = [
    ([10, 10, 10], [5]),
    ([8, 4, 0], [3]),
    ([10, 0, 0], [2]),
    ([0, 4, 10], [2]),
    ([0, 0, 0], [0]),

]
for inp, label in training_data:
    inp = convert_to_binary(inp, scale=10)
    label = convert_to_binary(label, scale=5)
    agent.next_state(INPUT=inp, LABEL=label, unsequenced=True)  # unsequenced is default; you can set it to `False` to run on data that is sequential





                                    # ----------- Inference on Content (using YouTube as an example) -----------#

yt_description = get_youtube_data("dQw4w9WgXcQ")

# Extracting features for input (using an LLM here)
llm_prompt = f"""
Analyze the following YouTube description: {yt_description}.

Provide a list of three numbers (1-10) representing:
1) Likelihood of re-posted content
2) Likelihood of compilation content
3) Likelihood of being an advertisement

Return only the three numbers as a list.
"""
extracted_features = ast.literal_eval(llm_call(llm_prompt))

# converting input to binary
input_to_agent = convert_to_binary(extracted_features, scale=10)

# # Initial prediction, predicting the likelihood of infringement based on the binary input
agent_response = agent.next_state(input_to_agent, unsequenced=True)
print("Agent raw binary response: ", agent_response)
print("Response percentage: ", sum(agent_response) / len(agent_response) * 100, "%")





                                    # ----------- Feedback Loop -----------#

# Closing the Learning Loop - passing feedback to the system to drive learning positively or negatively
res = input("Closing the Learning Loop-- was this input-pattern actually infringement (Y or N)?  ")
if res == "Y":
    agent.next_state(input_to_agent, LABEL=[1, 1, 1, 1, 1], unsequenced=True)
else:
    agent.next_state(input_to_agent, LABEL=[0, 0, 0, 0, 0], unsequenced=True)

# Re-evaluate After Feedback. To verify the learning, predict infringement again on the SAME input-pattern
agent_response = agent.next_state(input_to_agent, unsequenced=True)
print("Agent raw binary response: ", agent_response)
print("AFTER LEARNING LOOP, response percentage: ", sum(agent_response) / len(agent_response) * 100, "%")





                                      # ----------- Additional Test Input -----------#


# Testing with arbitrary inference calls (or include a LABEL argument to train)
agent_response = agent.next_state(convert_to_binary([0, 10, 10]), unsequenced=True) # try other input patterns
print("Agent raw binary response: ", agent_response)
print("ADDITIONAL TESTING, response percentage: ", sum(agent_response) / len(agent_response) * 100, "%")

## the ability of the agent to generalize depends on how often you train it
## you can check the agent.state to see how much it has been called into action