import openai
import config
import json
import csv
from datetime import datetime
import time

openai.api_key = config.api_key
json_output_file = open('entity_recognition_movies_output.json', 'w', newline='\r\n')

params = {
    "engine": "davinci",
    "max_tokens": 100,
    "temperature": 0.0,  #0.3 - Advanced tweet classifier (classification problem) | 0.0 - classification example | 
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


primerString = """I am a highly intelligent bot. If you give me a movie tite, I will give you information about the movie. If you give me a sentence that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

    Movie: Assault on Precinct 13
    Data: {"genre": "Action, Crime, Drama", "year": 2005, "director": "Jean-Francois Richet", "writer": "John Carpenter, James DeMonaco", "production_company": "Rogue Pictures"}

    Movie: Dead Meat
    Data: {"genre": "Comedy, Horror", "year": 2005, "director": "Philip Cruz", "writer": "Philip Cruz, James Hyde", "production_company": "Abyssmal Entertainment"}

    Movie: Cloudburst
    Data: {"genre": "Adventure, Comedy, Drama", "year": 2011, "director": "Thom Fitzgerald", "writer": "Thom Fitzgerald", "production_company": "Sidney Kimmel Entertainment"}

    Movie: Slaughterhouse
    Data: {"genre": "Comedy, Horror", "year": 1987, "director": "Rick Roessler", "writer": "Rick Roessler", "production_company": "American Artists"}

    Movie: And Then I Go
    Data: {"genre": "Drama", "year": 2017, "director": "Vincent Grashaw", "writer": "Brett Haley, Jim Shepard", "production_company": "Two Flints"}

    Movie: """

#print(primerString)

#print("******************")


with open('datasets/movie_input.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    counter = 0
    for row in spamreader:
        
        starttime = datetime.now().strftime("%f")
        prompt = primerString + row[0] + "\n    Data:"
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