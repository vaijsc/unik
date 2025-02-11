import json
from src.utils import read_jsonl

ASPECTS_SPACE = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
ASPECTS_AMASUM = ["general"]

def read_amasum(path):
    data = read_jsonl(path)
    data = [
        {
            "entity_id": d['entity_id'],
            "reviews": [
                {
                    "review_id": str(r['review_id']),
                    "review_text": ' '.join([s.strip() for s in  r['sentences']])
                } for r in d['reviews']
            ],
            "summaries": {
                "general": d['summaries'] 
            }
        } for d in data ]
    return data

def read_space(path):
    # data = read_jsonl(path)
    data = json.load(open(path))
    data = [
        {
            "entity_id": d['entity_id'],
            "reviews": [
                {
                    "review_id": str(r['review_id']),
                    "review_text": ' '.join([s.strip() for s in  r['sentences']])
                } for r in d['reviews']
            ],
            "summaries": d['summaries'],
        } for d in data ]
    return data

def read_data(path, dataset_name):
    data = []
    if dataset_name == "amasum":
        data = read_amasum(path)
    elif dataset_name == "space":
        data = read_space(path)
    assert len(data) > 0
    return data

def read_reviews(path, dataset_name):
    data = []
    if dataset_name == "amasum":
        data = read_amasum(path)
    elif dataset_name == "space":
        data = read_space(path)
    assert len(data) > 0
    reviews = []
    for d in data:
        for review in d['reviews']:
            reviews.append({"entity_id": d['entity_id'], **review})
    return reviews
