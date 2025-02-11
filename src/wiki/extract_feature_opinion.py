import json
import spacy
from src.utils import setup_logger, write_jsonl

logger = setup_logger(__name__, mode='basic')

class FeatureExtractor:
    def __init__(self, model="en_core_web_sm"):
        self.model = spacy.load(model)

    def get_feature_and_opinion(self, text):
        # Process the text using spaCy NLP pipeline
        doc = self.model(text)
        
        features = []
        opinions = []
        
        # Extract noun phrases (features) and adjectives (opinions)
        for chunk in doc.noun_chunks:
            filtered_chunk = ' '.join([token.lemma_ for token in chunk if token.pos_ not in {"ADJ", "DET", "PRON", "PUNCT"}])
            if filtered_chunk.strip() != "":
                features.append(filtered_chunk)
        
        for token in doc:
            if token.pos_ == "ADJ":  # Adjectives as opinions
                opinions.append(token.text)
        
        return {
            "text": text,
            "features": list(set(features)),  # Removing duplicates
            "opinions": list(set(opinions))
        }

def read_space(path):
    data = json.load(open(path))
    data = [
        {
            "entity_id": d['entity_id'],
            "reviews": [
                {
                    "review_id": str(r['review_id']),
                    "sentences": [s.strip() for s in r['sentences'] if s.strip() != "" ],
                } for r in d['reviews']
            ],
            "summaries": d['summaries'],
        } for d in data ]
    return data

def main():
    extractor = FeatureExtractor()

    data = read_space('data/raw/space/space_summ_test.json')
    extracted_data = []

    for d in data:
        entity_id = d['entity_id']
        review_data = []
        for review in d['reviews']:
            for sentence in review['sentences']:
                result = extractor.get_feature_and_opinion(sentence)
                review_data.append(result) 
        extracted_data.append({"entity_id": entity_id, "review_data": review_data})
        print(f"Entity: {entity_id} - Reviews: {len(review_data)}")

    write_jsonl(extracted_data, 'output/wiki/extracted_features.jsonl')

if __name__ == "__main__":
    main()
