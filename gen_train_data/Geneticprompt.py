import openai
import asyncio
from typing import List, Dict, Any
import argparse
import os 
from tqdm import trange, tqdm
import re 
import time
from utils import load_attributes
import numpy as np
import json 
import pandas as pd
from sampling import SampleManager

import random
import time


from openai import OpenAI




def shuffle_attributes(attributes):
    # 
    shuffled_attributes = attributes.copy()
    seed= int(time.time())
    random.seed(seed)  
    random.shuffle(shuffled_attributes)

    return shuffled_attributes


parser = argparse.ArgumentParser("")
parser.add_argument("--prompt", type=str, default="")
parser.add_argument("--parents", type=str, default="random")
parser.add_argument("--temperature", default=1, type=float, help="temp for generation")
parser.add_argument("--top_p", default=0.95, type=float, help="p value for generation")
parser.add_argument("--n_sample", default=10, type=int, help="the number of examples generated for each class")
parser.add_argument("--batch_size", default=20, type=int, help="")

parser.add_argument("--dataset", default='agnews', type=str, help="which dataset to synthesis")

parser.add_argument("--model_name", default='gpt-3.5-turbo', type=str, help="which model to use")
parser.add_argument("--model_type", default='demoprompt', type=str, help="which prompt type to use")
parser.add_argument("--max_tokens", default=1024, type=int, help="new length")
parser.add_argument("--output_dir", default='.', type=str, help="the folder for saving the generated text")
parser.add_argument("--input_dir", default='.', required=False, type=str, help="the folder for getting the input in-context learning text")

args = parser.parse_args()

if args.model_name=="deepseek-chat":
    args.api_key = ""
    args.openai_api_base= "https://api.deepseek.com"
elif args.model_name in ["gpt-3.5-turbo", "gpt-4o"]:
    args.api_key = ''
    args.openai_api_base= "https://api.openai.com/v1"
else:
    args.api_key= ""
    args.openai_api_base="http://localhost:8000/v1"
    
client = OpenAI(
    api_key=args.api_key,
    base_url=args.openai_api_base,
)


if args.dataset in ['chemprot']:
    args.entity = ['chemical', 'protein']
    args.label_names=['substrate','upregulator', 'downregulator', 'agonist', 'antagonist']
    args.attributes = ["length", "voice","sentence_structure" ,"interaction_verb","modifier","negation" ,"Entity Proximity"]
    
    """
    length:Length of the sentence
    
    voice:Whether the text is written in active or passive

    sentence_structure:Complexity of the sentence
    
    modifier: Adjectives or adverbs that modify the interaction, such as "slightly," "significantly," or "rarely," which can affect the strength or likelihood of interaction.
    negation: Whether the sentence includes negation
    
    interaction_verb: The verb that describes the interaction between the chemical and protein
    
    """
    args.label_def = {
        "upregulator": "the chemical Activates expression of the protein",
        "downregulator": "the chemical inhibits expression of the protein",
        "agonist": "the chemical triggering a biological response similar to the natural ligand",
        "antagonist": "the chemical diminishing its normal activity or interaction with its natural ligand.",
        #"product_of": "the protein is a product of the reaction on this chemical",
        "substrate": "the chemical is acted upon or modified by the protein, typically an enzyme",
        #"not": "There's no relation between the chemical and the protein from the generated sentence."
    }
    args.domain = 'Protein Chemical Relation extraction'
    args.input_dir = ''
    
    
elif args.dataset in ['ddi']:
    args.entity = ['drug', 'drug']
    args.domain = 'Drug-Drug Interaction extraction'
    args.attributes = ["length", "voice","polarity","interaction_verb","modifier","drug mentions","Entity Proximity"]
    """
    Drug Mentions: The specific names of drugs or substances and how they are referenced. Whether the text uses brand names, generic names, or even chemical names
    voice: Whether the text is written in active or passive
    modifier: Adjectives or adverbs that modify the interaction, such as "slightly," "significantly," or "rarely," which can affect the strength or likelihood of interaction.
    polarity: Whether the sentence conveys a positive, neutral, or negative interaction between the chemical and protein
    Entity Proximity: The distance between the two entities in the sentence
    """
    args.label_names=[ "effect","mechanism", "advise", "int"]
    args.label_def = {
        "mechanism":"One drug modifies the biological or pharmacological mechanism of another drug.",
        "effect":  "One drug alters the therapeutic or side effects of another drug.",
        "advise": "A recommendation or warning based on the interaction between two drugs, often advising dosage adjustment or avoidance.",
        "int": "A general interaction between two drugs is present, without specifying the exact nature or type of interaction."
     }
    args.input_dir = ''
    





elif args.dataset in ['semeval2010']:
    args.input_dir = ''
    args.attributes = ["length", "voice","Sentence Structure","Readability","Entity distance","domain"]
    args.domain="Relation Classification"
    args.label_names=["Cause-Effect","Component-Whole","Content-Container","Entity-Destination","Entity-Origin","Instrument-Agency","Member-Collection","Message-Topic","Product-Producer"]
    args.entity = ['entity', 'entity']
    args.label_def = {"Cause-Effect": "An entity causes an effect to another entity",
                      "Component-Whole": "An entity is a component of a whole",
                      "Entity-Destination": "An entity is transported to a destination",
                        "Product-Producer": "An entity is a product of a producer",
                        "Entity-Origin": "An entity is coming or derived from an origin",
                        "Member-Collection": "An entity is a member of a collection",
                        "Message-Topic": "An entity is a message topic",
                        "Content-Container": "An entity is a content of a container",
                        "Instrument-Agency": "An agent uses an instrument",}
    
elif args.dataset in ['conll04']:
    args.input_dir = ''
    args.attributes = ["length", "voice","Sentence Structure","Readability","Entity distance","domain"]
    args.label_names=["Kill","Live_In","Located_In","OrgBased_In","Work_For"]
    args.entity = ['entity', 'entity']
    args.domain="Relation Classification"
    args.label_def = {
        "Kill": "The entity kills another entity",
        "Live_In": "The person lives in a location",
        "Located_In": "The entity is located in a location",
        "OrgBased_In": "The organization is based in a location",
        "Work_For": "The person works for an organization"
    }
    
    """
    {
    "Kill": 0,
    "Live_In": 1,
    "Located_In": 2,
    "OrgBased_In": 3,
    "Work_For": 4
    }
    """



    
    
else:
    raise NotImplementedError





def main(args):

    label_names = args.label_names
    print(label_names)
    model = args.model_name
    openai.api_key = args.api_key
    
    df = pd.read_json(args.input_dir, lines=False) 
    
    for i, class_name in enumerate(label_names):

        print(i, class_name)
        label_def = args.label_def[class_name]
        #print(f"Prompt, Give a synthetic sample of {args.domain} about {re.sub('_', ' ', class_name)} following the requirements below")
        
        #we use sample manager to encode the samples from input_dir
        manager = SampleManager()
        #manager2 = SampleManager()
        
        
        
        
        filtered_df = df[df['_id'] == i]
        filtered_df = filtered_df.head(100)

        texts = filtered_df['text'].tolist()


        for _ in tqdm([1], desc="Adding samples"): 
            manager.add_samples(texts)        
            

        
        
        print(f"Class {i}: {class_name}, {len(texts)} samples.")
        
        
        diversity_scores=manager.calculate_diversity_scores()
        APS=diversity_scores['APS']
        INGF=diversity_scores['INGF']
        print(f"APS: {APS}, INGF: {INGF}")
        
        sent_cnt = 0 #
        attempt = 0 #          
        prompt_lst = []
        #examples = []


        for j in range(args.n_sample):
        
            if args.parents=="random":
                sample1=manager.get_random_sample()
                sample2=manager.get_random_sample()
            elif args.parents=="active":
                in_context_pair,distance = manager.find_least_similar_pair()
                sample1= manager.get_sample(in_context_pair[0])
                sample2= manager.get_sample(in_context_pair[1])
            else:
                print("need parents selection method")
                exit()

       

            if args.dataset ==  'chemprot':

                prompt_input =f"You need to generate synthetic data for the {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}> and <{args.entity[1]}> {args.entity[1]} </{args.entity[1]}> respectively in your response.
                                    The sentence should follow the requirements below: \n 
                                        1. The sentence must discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        5. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        6. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        7. The {re.sub('_', ' ', args.attributes[5])}' of the sentence should inherit from Example2;\n 
                                        8. The {re.sub('_', ' ', args.attributes[6])}' and entities must be different from the given 2 examples.
                                        """

                                
                
                
                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'                            
                                        
                                                                                        
                            
                

            elif args.dataset ==  'ddi':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                

                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between {args.entity[0]} and {args.entity[1]}. \n 
                                    The two {args.entity[0]}s should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. The sentence must discuss about the {args.entity[0]} and {args.entity[1]} with the relation {label_def}  \n 
                                        2. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        3. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        4. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        5. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        6. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        7. The {re.sub('_', ' ', args.attributes[5])}' of the sentence should inherit from Example2;\n 
                                        8. The {re.sub('_', ' ', args.attributes[6])}' and entities must be different from the given 2 examples.
                                        """



                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'
                
                
                
                
                
            elif args.dataset ==  'semeval2010':
                
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                




                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between 2 entities. \n 
                                    The '{class_name}' relation means {label_def}\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        5. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        6. The {re.sub('_', ' ', args.attributes[5])}' and entities must be different from the given 2 examples.
                                        """


                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'                    
            elif args.dataset ==  'conll04':
                prompt_input =f"You need to generate synthetic data for {args.domain} task. \n"
                in_context_input= f"Example 1: {sample1} \n Example 2: {sample2} \n" 
                



                Genetic_prompt = f"""
                                    Your task is to write a sentence about '{class_name}' relation between 2 entities. \n 
                                    The '{class_name}' relation means {label_def}\n
                                    The two entities should be marked with XML-style tags as <{args.entity[0]}> {args.entity[0]} </{args.entity[0]}>.
                                    The sentence should follow the requirements below: \n 
                                        1. The {re.sub('_', ' ', args.attributes[0])}' of the sentence should inherit from Example1;\n 
                                        2. The {re.sub('_', ' ', args.attributes[1])}' of the sentence should inherit from Example2;\n 
                                        3. The {re.sub('_', ' ', args.attributes[2])}' of the sentence should inherit from Example1;\n 
                                        4. The {re.sub('_', ' ', args.attributes[3])}' of the sentence should inherit from Example2;\n 
                                        5. The {re.sub('_', ' ', args.attributes[4])}' of the sentence should inherit from Example1;\n 
                                        6. The {re.sub('_', ' ', args.attributes[5])}' and entities must be different from the given 2 examples.
                                        """

                Genetic_prompt += f'Relation: {class_name}\n'
                Genetic_prompt += f'Text:'    


            else:
                raise NotImplementedError

            if attempt == 0 and len(prompt_lst) == 0:
                print(f"Prompt Input: {prompt_input}")

            prompt_lst.append([{"role": "user", "content": prompt_input+in_context_input+Genetic_prompt}])
            
            
            if len(prompt_lst) == args.batch_size:  #
                # 
                args.attributes = shuffle_attributes(args.attributes)
                try:
                    attempt += 1
                    # 
                    return_msg = [
                        client.chat.completions.create(
                            model=model,
                            messages=prompt,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            top_p=args.top_p,
                        ).choices[0].message.content
                        for prompt in prompt_lst
                    ]
                    assert len(return_msg) == args.batch_size
                    valid = 0
                    tmp = []
                    for msg in return_msg:
                        if "I apologize"  in msg or  "sorry"  in msg or  "Sorry" in msg or  "an AI language model" in msg or "I cannot perform" in msg:
                            continue
                        else:
                            #print(clean_str(msg))
                            valid += 1
                            example = {"_id": i, "text": msg}
                            #example.update(attr) #add randomly selected attrs to dic
                            #examples.append( example)
                            tmp.append(example)

                    manager.add_samples([x['text'] for x in tmp])

                    
                    sent_cnt += valid 
                    prompt_lst = []
                    #attr_lst = []
                    
                    
                    print(f"CLass {i}: {class_name}, Attempt: {attempt}, Sent cnt: {sent_cnt}. ")
                    prefix = f"{args.dataset}/{class_name}/train_p{args.top_p}_{i}_{attempt}.jsonl"
                    
                    os.makedirs(f"{args.output_dir}/{args.dataset}/{class_name}", exist_ok= True)
                    f = open(f"{args.output_dir}/{prefix}", 'w')
                    
                    for e in tmp:
                        f.write(json.dumps(e) + "\n")
                    f.close()
                    
                except json.decoder.JSONDecodeError:
                    print(f"JSONDecodeError! Received empty response. Attempt: {attempt}")
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)  
                    continue
                except openai.error.RateLimitError:
                    print("Rate Limit Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue
                except  openai.error.APIError:
                    print("API Error! Attempt:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue
                except openai.error.APIConnectionError:
                    print("APIConnectionError", attempt)
                    prompt_lst = []
                    attr_lst = []
                    time.sleep(5)
                    continue 
                except openai.error.InvalidRequestError:
                    print("InvalidRequestError! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
                except openai.error.Timeout:
                    print("Timeout Error! Invalid Request:", attempt)
                    prompt_lst = []
                    attr_lst = []
                    continue
            #if (sent_cnt >= 1200 and synthetic_APS>=APS and synthetic_INGF>=INGF ) or sent_cnt >= 3000 or attempt > 200:
            if  sent_cnt >= args.n_sample or attempt > 1000:
                break
       

if __name__ == '__main__':
    main(args)

