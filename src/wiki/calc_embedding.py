import pickle
from src.wiki.embedding import EmbeddingModel
from src.utils import read_jsonl, write_jsonl

if __name__ == "__main__":
    model_path = '/lustre/scratch/client/vinai/users/hoangnv49/hf/BAAI/bge-base-en-v1.5'
    encoder = EmbeddingModel(model_path)

    input_path = "output/wiki/matched_features.jsonl"

    # create aspect embeddings
    ASPECTS = ['building', 'cleanliness', 'food', 'location', 'rooms', 'service']
    aspects = {
        "building": "Building: structure, typically with a roof and walls, standing more or less permanently in one place",
        "cleanliness": "cleanliness: abstract state of being clean and free from dirt" ,
        "food": "Food :  any substance consumed to provide nutritional support for the body; form of energy stored in chemical",
        "location": "location: location of the object, structure or event",
        "rooms": "Room: distinguishable space for humans within an enclosed structure",
        "service": "service: economic product that directly satisfies wants without producing a lasting asset", 
    }
    embeddings = encoder.encode([aspects[a] for a in ASPECTS], max_length=128)
    aspect_embeddings = {"aspects": ASPECTS, "embeddings": embeddings}
    pickle.dump(aspect_embeddings, open('output/wiki/aspect_embeddings.pkl', 'wb'))

    # create feature embeddings 
    entity_data = [] 
    for entity in read_jsonl(input_path):
        review_data = []
        for review in entity['review_data']:
            text = review['text']
            for feature in review['features']:
                for opinion in review['opinions']:
                    review_data.append({"feature": feature, "opinion": opinion, "text": text, "prompt": f"{feature} {opinion} {text}"})
        embeddings = encoder.encode([d["prompt"] for d in review_data], max_length=128)
        for i, d in enumerate(review_data):
            review_data[i]['embedding'] = embeddings[i]
        entity_data.append({"entity_id": entity['entity_id'], "features": review_data})

    # save to file
    pickle.dump(entity_data, open('output/wiki/feature_embeddings.pkl', 'wb'))
    
