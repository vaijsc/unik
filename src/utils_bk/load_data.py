import csv
import json
import os
import yaml
from collections import defaultdict

def load_config(file_path):
    with open(file_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
            return config
        except yaml.YAMLError as exc:
            print(f"Error reading YAML file: {exc}")
            raise Exception

def load_jsonl(path, n_examples=0):
    print(f"Load data from {path}")
    data = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line != "":
                data.append(json.loads(line))
                if n_examples > 0 and len(data) >= n_examples:
                    break
    return data

def get_n_processed_examples(path):
    n_processed_examples = 0
    if os.path.exists(path):
        processed_examples = load_jsonl(args.output_path)
        n_processed_examples = len(processed_examples)
    print(f"Found {n_processed_examples} processed examples")
    return n_processed_examples

def load_splits_data(file_path):
    """
    Input: 
        file_path (str): The path to the file containing the space slits data.
    Output: 
        data_dict (dict): A dictionary with keys 'dev' and 'test', each containing a list of IDs.
    """
    data_dict = {}
    with open(file_path, 'r') as file:
        for line in file:
            parts = line.strip().split('\t')
            id = parts[0]
            data_type = parts[1]
            data_dict[id] = data_type
    print(f"Load splits from {file_path}")
    # print(f"Len dev:", len([k for k,v in data_dict.items() if v == 'dev']))  
    # print(f"Len test:", len([k for k,v in data_dict.items() if v == 'test']))  
    return data_dict

def load_train_data(file_path):
    """
    Input: 
        file_path (str): The path to the file containing the train data.
        dataset_name (str): The name of the dataset ('space' or 'amazon').
    Output: 
        entities (list): A list of lists, where each inner list contains reviews for an entity.
    """
    print(f"Load train data from {file_path}")
    with open(file_path, 'r') as file:
        data = json.load(file)

    entities = []
    for entity in data:
        reviews = []
        for review in entity['reviews']:
            reviews.append(review['sentences'])
        entities.append(reviews)
    print(f"Len data:", len(entities)) 
    return entities

def load_summ_data(file_path, splits_path, aspect_name='general'):
    """
    Input: 
        file_path (str): The path to the file containing the test data.
        splits_path (str): The path to the file containing splits infor.
        aspect_name (str): The name of the aspect to be used for labels.
    
    Output: 
        entities (list): A list of dictionaries, each containing 'reviews' and 'labels' for an entity.
    """
    if splits_path:
        print(f"Load summ data from {file_path}, aspect={aspect_name}")
        splits = load_splits_data(splits_path)
        with open(file_path, 'r') as file:
            data = json.load(file)

        dev_entities = []
        test_entities = []
        for entity in data:
            reviews = []
            for review in entity['reviews']:
                reviews.append(review['sentences'])
            try:
                labels = entity['summaries'][aspect_name]
            except:
                breakpoint()
            if splits[entity['entity_id']] == 'dev':
                dev_entities.append({'reviews': reviews, 'labels': labels})
            elif splits[entity['entity_id']] == 'test':
                test_entities.append({'reviews': reviews, 'labels': labels})
        print(f"Len dev:", len(dev_entities), "| Len test:", len(test_entities))
        return dev_entities, test_entities
    else:
        print(f"Load amasum data from {file_path}")
        data = []
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))

        test_entities = []
        for entity in data:
            categories = entity['categories']
            reviews = []
            for review in entity['reviews']:
                reviews.append(review['sentences'])
            try:
                labels = entity['summaries']
            except:
                breakpoint()
            test_entities.append({'reviews': reviews, 'labels': labels, 'categories': categories})
        print("| Len test:", len(test_entities))
        return 0, test_entities

def get_label_from_llms(response, aspect_names):
    response = response.split('(')[0]
    response = response.split('\n')
    aspect_list = []
    for line in response:
        if "Aspect:" in line:
            aspects = line.strip().split(':')
            aspect_list = aspects[1].split(',')
    labels = []
    for item in aspect_list:
        item = item.strip().strip('.')
        if item in aspect_names:
            labels.append(item)
        else:
            labels.append('general')
    return labels

# def get_knowledge_from_llms(file_path):
#     INTERESTED_ASPECTS = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
#     METADATA_KEYS = ['aspects', 'features', 'expressions', 'description']
#     data = load_jsonl(file_path)    
#     # ps = nltk.stem.porter.PorterStemmer() # stemming e.g. ps.stem('words')
#     entities = defaultdict(list)
#     # knowledge = []
#     cnt_invalid = 0
#     for review in data:
#         entity_idx, review_idx = review['entity_idx'], review['review_idx']
#         responses = [r.strip() for r in review['response'].split("\n") if r.strip().startswith('#')]
#         for response in responses:
#             data_fields = [r for r in response.split("#") if r.strip() != '']
#             metadata = {}
#             for field in data_fields:
#                 try:
#                     key, value = tuple(field.split(':', 1))
#                     key, value = key.strip(), value.strip()
#                 except:
#                     continue
#                 if "Aspect name" in key:
#                     aspects = [a.strip().lower() for a in value.split(',') if a.strip().lower() != 'none']
#                     # if aspect in INTERESTED_ASPECTS then keep
#                     # if aspect == none skip
#                     # else move to "other"
#                     metadata['aspects'] = [a if a in INTERESTED_ASPECTS else 'other' for a in aspects]
#                     continue
#                 if "Entity name" in key:
#                     # discard 
#                     metadata['features'] = [f.strip().lower() for f in value.split(',') if f.strip().lower() != 'none']
#                     # features = field.split(":")[1].strip()
#                     # metadata['features'] = [item.lower().strip() for item in features.split(",") if item]
#                     continue
#                 if "Expression phrases" in key:
#                     metadata['expressions'] = [e.strip().lower() for e in value.split(',') if e.strip().lower() != 'none']
#                     # expressions = field.split(":")[1].strip()
#                     # metadata['expressions'] = [item.lower().strip() for item in expressions.split(",") if item]
#                     continue
#                 if "Description" in key:
#                     metadata['description'] = value if value != "none" else ""
#                     # metadata['description'] = field.split(":")[1].strip()
#                     continue
#             if metadata:
#                 if len(metadata) != len(METADATA_KEYS):
#                     # print(metadata)
#                     cnt_invalid += 1
#                     continue
#                 for v in metadata.values():
#                     if (isinstance(v, list) and len(v) == 0) or (isinstance(v, str) and v == ''):
#                         # print(metadata)
#                         cnt_invalid += 1
#                         continue
#                 # metadata['entity_idx'] = entity_idx
#                 metadata['review_idx'] = review_idx
#                 entities[entity_idx] = entities.get(entity_idx, []) + [metadata]
#                 # knowledge.append(metadata)
#     print(f"Discarded {cnt_invalid} invalid infos")
#     return entities

def get_knowledge_from_llms(file_path):
    INTERESTED_ASPECTS = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
    METADATA_KEYS = ['aspects', 'features', 'expressions', 'description']
    data = load_jsonl(file_path)    
    # ps = nltk.stem.porter.PorterStemmer() # stemming e.g. ps.stem('words')
    entities = defaultdict(lambda: defaultdict(list))
    # knowledge = []
    cnt_invalid = 0
    for review in data:
        entity_idx, review_idx = review['entity_idx'], review['review_idx']
        responses = [r.strip() for r in review['response'].split("\n") if r.strip().startswith('#')]
        for response in responses:
            data_fields = [r for r in response.split("#") if r.strip() != '']
            metadata = {}
            for field in data_fields:
                try:
                    key, value = tuple(field.split(':', 1))
                    key, value = key.strip(), value.strip()
                    if key == "" or value == "":
                        continue
                except:
                    continue
                if "Aspect name" in key:
                    aspects = [a.strip().lower() for a in value.split(',') if a.strip().lower() != 'none']
                    # if aspect in INTERESTED_ASPECTS then keep
                    # if aspect == none skip
                    # else move to "other"
                    metadata['aspects'] = [a if a in INTERESTED_ASPECTS else 'other' for a in aspects]
                    continue
                if "Entity name" in key:
                    # discard 
                    # metadata['features'] = [f.strip().lower() for f in value.split(',') if f.strip().lower() != 'none']
                    
                    # keep as one
                    metadata['features'] = [value.strip().lower()] if value.strip().lower() != 'none' else []
                    continue
                if "Expression phrases" in key:
                    # metadata['expressions'] = [e.strip().lower() for e in value.split(',') if e.strip().lower() != 'none']

                    # keep as one
                    metadata['expressions'] = [value.strip().lower()] if value.strip().lower() != 'none' else []
                    continue
                if "Description" in key:
                    metadata['description'] = value if value != "none" else ""
                    # metadata['description'] = field.split(":")[1].strip()
                    continue
            if metadata:
                if len(metadata) != len(METADATA_KEYS):
                    # print(metadata)
                    cnt_invalid += 1
                    continue
                for v in metadata.values():
                    if (isinstance(v, list) and len(v) == 0) or (isinstance(v, str) and v == ''):
                        # print(metadata)
                        cnt_invalid += 1
                        continue
                if metadata['features'][0].lower() not in metadata['description'].lower():
                    metadata['aspects'] = ['general']
                entities[entity_idx][review_idx] += [metadata]
    print(f"Discarded {cnt_invalid} invalid infos")
    return entities
                          
if __name__ == "__main__":
    path_file = "method/data/space_test_multi.jsonl"
    data = get_knowledge_from_llms(path_file)
    from pprint import pprint  
    # breakpoint()
    print(len(data[0]))
    pprint(data[0][0], width=120)
