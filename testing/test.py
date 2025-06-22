from openai import OpenAI
import time



client1=OpenAI(
    api_key="xai-pyE9GxDil42uj8sKRLFLgMh6g85fFl3ipwMSgKaJ3BWOYNOjdt8WnMxydtXXj6fj0Z1N4IFrEnrP5QoU",
    base_url="https://api.x.ai/v1" ,
)
completion=client1.chat.completions.create(
    model="grok3" ,
    messages=[
        {"role":"user","content" : "What is cinematography"}

    ]
)


