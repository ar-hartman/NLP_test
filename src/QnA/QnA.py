import openai
import config
import json
import csv
from datetime import datetime
import time

openai.api_key = config.api_key
json_output_file = open('QnA_output.json', 'w', newline='\r\n')

params = {
    "engine": "davinci",
    "max_tokens": 100,
    "temperature": 0.0625,  #deviation from defaults
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "best_of": 1,
    "stream": 'false',
    "logprobs": 'null',
    "stop": "\n"
}


##good at: complex intent, cause and effect, summarization for audience
#openai.Completion.create(engine="davinci")

##good at: language translation, complex classification, text sentiment, summarization
#openai.Completion.create(engine="curie")

##good at: moderate classification, semantic search classification
#openai.Completion.create(engine="babbage")

##good at: parsing text, simple classification, address correction, keywords
#openai.Completion.create(engine="ada")


primerString = """I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

    Q: What is human life expectancy in the United States?
    A: Human life expectancy in the United States is 78 years.

    Q: Who was president of the United States in 1955?
    A: Dwight D. Eisenhower was president of the United States in 1955.

    Q: Which party did he belong to?
    A: He belonged to the Republican Party.

    Q: What is the square root of banana?
    A: Unknown

    Q: How does a telescope work?
    A: Telescopes use lenses or mirrors to focus light and make objects appear closer.

    Q: Where were the 1992 Olympics held?
    A: The 1992 Olympics were held in Barcelona, Spain.

    Q: How many squigs are in a bonk?
    A: Unknown

    Q: """

#print(primerString)

#print("******************")


with open('datasets/AXA-question-bank.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    counter = 0
    for row in spamreader:
        
        starttime = datetime.now().strftime("%f")
        prompt = primerString + row[0] + "\n    A:"
        #print(prompt)
    
        response = openai.Completion.create(engine=params['engine'], prompt=prompt, max_tokens=params['max_tokens'], 
            temperature=params['temperature'], top_p=params['top_p'], frequency_penalty=params['frequency_penalty'], 
            presence_penalty=params['presence_penalty'], stop=params['stop'])


        endTime = datetime.now().strftime("%f")

        latency = (float(endTime) / 1000) - (float(starttime) / 1000)
        #latency.microseconds
        
        #print("latency: " + str(latency))

        record = {"id": counter,
            "query": row[0],
            "prompt_passed": prompt,
            "response": response['choices'][0]['text'],
            "latency": str(latency)
            }

        json.dump(record, json_output_file)
        json_output_file.write("\n")


        counter = counter + 1
        time.sleep(2)
