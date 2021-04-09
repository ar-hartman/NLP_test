import openai
import config
import json
import csv
from datetime import datetime
import time

openai.api_key = config.api_key
json_output_file = open('entity_recognition_output.json', 'w', newline='\r\n')

params = {
    "engine": "davinci",
    "max_tokens": 500,
    "temperature": 0.30,  #0.3 - Advanced tweet classifier (classification problem) | 0.0 - classification example | 
    "top_p": 1,
    "frequency_penalty": 0,
    "presence_penalty": 0,
    "n": 1,
    "best_of": 1,
    "stream": 'false',
    "logprobs": 'null',
    "stop": ";"
}


##good at: complex intent, cause and effect, summarization for audience
#openai.Completion.create(engine="davinci")

##good at: language translation, complex classification, text sentiment, summarization
#openai.Completion.create(engine="curie")

##good at: moderate classification, semantic search classification
#openai.Completion.create(engine="babbage")

##good at: parsing text, simple classification, address correction, keywords
#openai.Completion.create(engine="ada")


primerString = """I am a highly intelligent bot. If you give me a sentence, I will give you all of the entities. If you give me a sentence that is nonsense, trickery, or has no clear answer, I will respond with "Unknown".

    S: Thousands of demonstrators have marched through London to protest the war in Iraq and demand the withdrawal of British troops from that country.
    R: {"Entities": [{"BeginOffset": 0, "EndOffset": 26, "Text": "Thousands of demonstrators", "Type": "QUANTITY"}, {"BeginOffset": 48, "EndOffset": 54, "Text": "London", "Type": "LOCATION"}, {"BeginOffset": 77, "EndOffset": 81, "Text": "Iraq", "Type": "LOCATION"}, {"BeginOffset": 111, "EndOffset": 118, "Text": "British", "Type": "OTHER"}]};

    S: Families of soldiers killed in the conflict joined the protesters who carried banners with such slogans as "Bush Number One Terrorist" and "Stop the Bombings."
    R: {"Entities": [{"BeginOffset": 111, "EndOffset": 115, "Text": "Bush", "Type": "PERSON"}, {"BeginOffset": 116, "EndOffset": 126, "Text": "Number One", "Type": "OTHER"}]};

    S: They marched from the Houses of Parliament to a rally in Hyde Park.
    R: {"Entities": [{"BeginOffset": 22, "EndOffset": 42, "Text": "Houses of Parliament", "Type": "LOCATION"}, {"BeginOffset": 57, "EndOffset": 66, "Text": "Hyde Park", "Type": "LOCATION"}]};

    S: Police put the number of marchers at 10000 while organizers claimed it was 1,000,000.
    R: {"Entities": [{"BeginOffset": 38, "EndOffset": 43, "Text": "10000", "Type": "QUANTITY"}, {"BeginOffset": 76, "EndOffset": 85, "Text": "1,000,000", "Type": "QUANTITY"}]};

    S: Internet retailer Amazon.com raised more than $ 4.5 million from its customers for the Red Cross.
    R: {"Entities": [{"BeginOffset": 18, "EndOffset": 28, "Text": "Amazon.com", "Type": "ORGANIZATION"}, {"BeginOffset": 36, "EndOffset": 59, "Text": "more than $ 4.5 million", "Type": "QUANTITY"}, {"BeginOffset": 87, "EndOffset": 96, "Text": "Red Cross", "Type": "ORGANIZATION"}]};

    S: """

#print(primerString)

#print("******************")


with open('datasets/ent_rec_aws.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile)
    counter = 0
    for row in spamreader:
        
        starttime = datetime.now().strftime("%f")
        prompt = primerString + row[0] + "\n    R:"
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
        time.sleep(2)
