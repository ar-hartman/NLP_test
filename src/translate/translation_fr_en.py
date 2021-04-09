import openai
import config
import json
import csv
from datetime import datetime
import time

openai.api_key = config.api_key
json_output_file = open('translate_fr_en (temp = 0.75).json', 'w', newline='\r\n')

params = {
    "engine": "davinci",
    "max_tokens": 100,
    "temperature": 0.75,  
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


primerString = """Je suis un robot très intelligent. Si vous me donnez une phrase, je la traduirai pour vous. Si vous me donnez une phrase qui n'a aucun sens, une supercherie ou qui n'a pas de réponse claire, je répondrai par "Inconnu".

    French: Où se trouvent les toilettes?
    English: Where is the bathroom?

    French: Combien coûte le poisson?
    English: How much does the fish cost?

    French: Pouvez-vous m'aider avec cet article?
    English: Can you help me with this item?

    French: Nous parlerons plus tard.
    English: We will speak later.

    French: C’est un sentiment compliqué, un regret.
    English: It's a complicated feeling, regret.

    French: """

#print(primerString)

#print("******************")


with open('datasets/fr-en.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    counter = 0
    for row in spamreader:
        
        starttime = datetime.now().strftime("%f")
        prompt = primerString + row[0] + "\n    English:"
        #print(prompt)
        #print("******************")

    
        response = openai.Completion.create(engine=params['engine'], prompt=prompt, max_tokens=params['max_tokens'], 
            temperature=params['temperature'], top_p=params['top_p'], frequency_penalty=params['frequency_penalty'], 
            presence_penalty=params['presence_penalty'], stop=params['stop'])


        endTime = datetime.now().strftime("%f")

        latency = (float(endTime) / 1000) - (float(starttime) / 1000)
        # latency.microseconds
        
        # print("latency: " + str(latency))

        record = {"id": counter,
            "query": row[0],
            "prompt_passed": prompt,
            "response": response['choices'][0]['text'],
            "latency": str(latency)
            }

        json.dump(record, json_output_file)
        json_output_file.write("\n")


        counter = counter + 1
        time.sleep(1)
