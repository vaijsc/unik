import pygtrie 
import json
import pickle
from src.utils import setup_logger, read_jsonl, write_jsonl

logger = setup_logger(__name__, mode='basic')


def build_trie(input_path, max_len):
    entities = [json.loads(line) for line in open(input_path)] 
    entities = [e for e in entities if len(e['label']) <= max_len]
    logger.info(f"Read {len(entities)} entities from `{input_path}`")

    tree = pygtrie.CharTrie()
    for entity in entities:
        tree[entity['label']] = entity 
    return tree

def is_valid(feature):
    for c in "~!?[]{}<>|@#$%^&":
        if c in feature:
            return False
    return True

if __name__ == "__main__":
    wiki_entity_path = 'output/wiki/entities.jsonl'
    input_path = 'output/wiki/extracted_features.jsonl'
    output_path = 'output/wiki/matched_features.jsonl' 

    # Find max length of feature
    max_len = 0
    for entity in read_jsonl(input_path):
        for review in entity['review_data']:
            for feature in review['features']:
                if len(feature) > max_len:
                    max_len = len(feature)

    tree = build_trie(wiki_entity_path, max_len)

    matched_entity_data = []
    for entity in read_jsonl(input_path):
        entity_id = entity['entity_id']
        matched_review_data = []
        total_entities = 0
        total_matched_entities = 0
        for review in entity['review_data']:
            matched_features = []
            total_entities += len(review['features']) 
            if len(review['opinions']) == 0:
                continue
            for feature in review['features']:
                if not is_valid(feature):
                    continue
                if tree.has_key(feature):
                    matched_features.append({"feature": feature, "desc": tree[feature]['desc']})
                    total_matched_entities += 1
                if len(feature) > max_len:
                    print(feature)
                    max_len = len(feature)
            if matched_features and len(review["opinions"]) > 0:
                matched_review_data.append({"text": review['text'], "features": matched_features, "opinions": review['opinions']})
        print(f"Entity: {entity_id}")
        print(f"# Sentences have Wiki entities: {len(matched_review_data)}/{len(entity['review_data'])}")
        print(f"Matched entities {total_matched_entities}/{total_entities}")

        entity['review_data'] = matched_review_data
        matched_entity_data.append(entity)

    write_jsonl(matched_entity_data, output_path)
