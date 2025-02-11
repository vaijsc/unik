import pickle
import torch
from src.utils import read_jsonl, write_jsonl, setup_logger
from vllm import LLM, SamplingParams

ASPECTS = ["building", "cleanliness", "food", "location", "rooms", "service"]

class Graph():
    def __init__(self, entity_id):
        self._entity_id = entity_id
        self.aspects = {
            "building": {},
            "cleanliness": {},
            "food": {},
            "location": {},
            "rooms": {},
            "service": {},
        }
    
    def add(self, aspect, feature, opinion, text):
        if aspect not in self.aspects:
            raise ValueError(f"Aspect {aspect} is not valid")

        if feature not in self.aspects[aspect]:
            self.aspects[aspect][feature] = {
                "count": 0,
                "opinions": {}
            }
        if opinion not in self.aspects[aspect][feature]['opinions']:
            self.aspects[aspect][feature]['opinions'][opinion] = {
                "texts": []
            }
        self.aspects[aspect][feature]['count'] += 1
        self.aspects[aspect][feature]['opinions'][opinion]['texts'].append(text)

    def get(self, aspect, n_feature, n_opinion):
        if aspect not in self.aspects:
            raise ValueError(f"Aspect {aspect} is not valid")
        features = sorted(self.aspects[aspect].items(), key=lambda x: x[1]['count'], reverse=True)
        for feature, data in features[:n_feature]:
            opinions = sorted(data['opinions'].items(), key=lambda x: len(x[1]['texts']), reverse=True)
            for opinion, data in opinions[:n_opinion]:
                texts = data['texts'][0]
                yield feature, opinion, texts

def summ(model, prompt, max_tokens=128, temperature=1.0, top_p=0.9, top_k=50):
    params = SamplingParams(max_tokens=max_tokens, temperature=temperature, top_p=top_p, top_k=top_k) 
    output = model.generate(prompt, sampling_params=params)
    return output[0].outputs[0].text.strip()

if __name__ == "__main__":
    feature_embeddings = pickle.load(open('output/wiki/feature_embeddings.pkl', 'rb'))
    aspect_embeddings = pickle.load(open('output/wiki/aspect_embeddings.pkl', 'rb'))

    aspects = aspect_embeddings['aspects']
    embs = torch.tensor(aspect_embeddings['embeddings'], device='cuda') 

    model_path = '/lustre/scratch/client/vinai/users/hoangnv49/hf/mistralai/Mistral-7B-Instruct-v0.2'
    model = LLM(model_path)

    output = []
    for entity in feature_embeddings:
        entity_id = entity['entity_id']
        graph = Graph(entity_id)
        for review_data in entity['features']:
            review_emb = torch.tensor(review_data['embedding'], device='cuda')
            scores = torch.matmul(embs, review_emb)
            # take aspect with highest score
            idx = torch.argmax(scores, dim=0)
            aspect = aspects[idx]
            if scores[idx].item() > 0.4:
                graph.add(aspect, review_data['feature']['feature'], review_data['opinion'], review_data['text'])
        # collect information to summarize
        general_reviews = []
        for aspect in ASPECTS:
            reviews = []
            for feature, opinion, text in graph.get(aspect, 2, 2):
                reviews.append(f"{feature} {opinion} {text}")
                general_reviews.append(f"{feature} {opinion} {text}")
            prompt = " ".join(reviews)
            prompt = f"Briefly summarize the following reviews for aspect {aspect}:\n{prompt}"
            summary = summ(model, prompt)
            output.append({"entity_id": entity_id, "aspect": aspect, "summaries": [summary]})

        # general aspect
        prompt = " ".join(general_reviews)
        prompt = f"Briefly summarize the following reviews:\n{prompt}"
        summary = summ(model, prompt, max_tokens=256)
        output.append({"entity_id": entity_id, "aspect": "general", "summaries": [summary]})

    write_jsonl(output, 'output/wiki/summ.jsonl')
