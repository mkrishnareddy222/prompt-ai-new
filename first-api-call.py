#step 1
# Setup And installation
#import subprocess
#subprocess.run(['pip', 'install', 'groq', '-q'])
#print('✅ Groq installed')

# Define API key 
import os

import config

GROQ_API_KEY = config.GROQ_API_KEY
MODEL = config.MODEL
MAX_TOKENS = config.MAX_TOKENS
TEMPERATURE = config.TEMPERATURE
print('✅ Groq API key set')


# Step 3: Create the Groq client
from groq import Groq
client = Groq(api_key=GROQ_API_KEY)
print(f'✅ Client ready | Model: {MODEL}')




response =  client.chat.completions.create(
    model=MODEL,
    max_tokens=MAX_TOKENS,
    temperature=TEMPERATURE,
    messages=[{
        "role": "user",
      "content" : "What are the top 3 Sports in 2026"
    }
    ]
)


print("AI RESPONSE")
print(response.choices[0].message.content)