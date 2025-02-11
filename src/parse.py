import argparse
# from nltk.stem import PorterStemmer
# from nltk.corpus import stopwords
# from nltk.tokenize import word_tokenize
import logging
import re
from tqdm import tqdm
from src.utils import read_jsonl, setup_logger, write_jsonl

logger = logging.getLogger(__name__)


ASPECTS_SPACE = ["general", "rooms", "location", "service", "cleanliness", "building", "food"]
ASPECTS_AMASUM = ["general"]


def is_valid(text):
    skipped_words = set(["none", "n/a"])
    text = text.lower().strip()
    if text in skipped_words:
        return False
    if text.startswith('['):
        return False
    if text.startswith('n/a'):
        return False
    if text.startswith('-'):
        return True
    # if text.startswith('hotel'):
        # return False
    return True

# def is_valid_amasum(text):
#     skipped_words = set(["none", "n/a"])
#     text = text.lower().strip()
#     if text in skipped_words:
#         return False
#     if text.startswith('['):
#         return False
#     if text.startswith('-'):
#         return False
#     if text.startswith('n/a'):
#         return False
#     return True


def parse_aspect_space(text):
    text = text.strip().lower()
    if ',' in text:
        text = text.split(',', 1)[0]

    a_room = set(['room', 'bathroom', 'bed', 'beds', 'bath', 'accommodations', 'tv', 
                  'television', 'accommodation', 'bathrooms', 'bedrooms', 'tvs', 'wifi', 'wi-fi'])
    a_location = set(['view'])
    a_service = set(['staff', 'services', 'waiters'])
    a_cleanliness = set('hygiene')
    a_building = set(['gym', 'pool', 'amenities', 'buildings']) 
    a_food = set(['breakfast', 'drink', 'drinks'])

    if not is_valid(text):
        return None
    if text in a_room:
        return 'rooms'
    if text in a_location:
        return 'location'
    if text in a_service:
        return 'service'
    if text in a_cleanliness:
        return 'cleanliness'
    if text in a_building:
        return 'building'
    if text in a_food:
        return 'food'
    if text in ASPECTS_SPACE:
        return text
    return 'general'

def parse_aspect_amasum(text):
    text = text.strip().lower()
    if ',' in text:
        text = text.split(',', 1)[0]
    if not is_valid(text):
        return None
    if text in ASPECTS_AMASUM:
        return text
    return 'general'

def parse_feature(text):
    features = []
    for f in text.lower().split(','):
        f = f.strip()
        if is_valid(f):
            features.append(f)
    return features

def parse_opinion(text):
    opinions = []
    for o in text.lower().split(','):
        o = o.strip()
        if is_valid(o):
            opinions.append(o)
    return opinions

def parse_sentence_space(text):
    """
    example text: "#Aspect name: Location #Entity name: Hotel Navona #Expression phrases: perfect, beautiful #Description: The Hotel Navona is situated in perfect location to take advantage of Rome's beautiful sights.
    
    Extract Aspect, Feature, Opinion, Description from the text
    """
    pattern = r'#Aspect name:\s*(.*?)\s*#Entity name:\s*(.*?)\s*#Expression phrases:\s*(.*?)\s*#Description:\s*(.*)'  
    # matches = re.findall(pattern, text, re.DOTALL)
    matches = re.findall(pattern, text)
    metadata = {}
    if matches:
        aspect_text, feature_text, opinion_text, description_text = matches[0]

        aspect = parse_aspect_space(aspect_text)
        feature = feature_text.lower().strip() if is_valid(feature_text) else None
        opinion = parse_opinion(opinion_text)
        description = description_text if is_valid(description_text) else None

        # validate data 
        if aspect is None or feature is None or len(opinion) == 0 or description is None:
            return None

        if feature not in description.lower():
            aspect = ['general']
        metadata['aspect'] = aspect
        metadata['feature'] = feature
        metadata['opinions'] = opinion
        metadata['description'] = description

    return metadata

def parse_sentence_amasum(text):
    """ 
    example text: #Aspect name: Functionality, TV #Entity name: Remote control #Opinion phrases: love, extend, long HDMI cord #Description: I love that the remote control allows me to extend the TV down from a high position using a long HDMI cord.\n#Aspect name: Functionality, TV #Entity name: Closed captioning #Opinion phrases: drives nutty, have to tell, every time, every episode #Description: I'm driven nutty by the fact that I have to tell the TV to turn on closed captioning every time and for every episode on Hulu.

    Extract Aspect, Feature, Opinion, Description from the text
    """
    pattern = r'#Aspect name:\s*(.*?)\s*#Entity name:\s*(.*?)\s*#Opinion phrases:\s*(.*?)\s*#Description:\s*(.*)'  
    matches = re.findall(pattern, text, re.DOTALL)
    # matches = re.findall(pattern, text)
    metadata = {}
    if matches:
        aspect_text, feature_text, opinion_text, description_text = matches[0]

        aspect = parse_aspect_amasum(aspect_text)
        feature = feature_text.lower().strip() if is_valid(feature_text) else None
        opinion = parse_opinion(opinion_text)
        description = description_text if is_valid(description_text) else None

        # validate data 
        if aspect is None or feature is None or len(opinion) == 0 or description is None:
            return None

        if feature not in description.lower():
            aspect = ['general']
        metadata['aspect'] = aspect
        metadata['feature'] = feature
        metadata['opinions'] = opinion
        metadata['description'] = description

    return metadata


def parse_extraction_output(file_path, dataset_name):
    parse_function = None
    if dataset_name == 'space':
        parse_function = parse_sentence_space 
    elif dataset_name == "amasum":
        parse_function = parse_sentence_amasum
    else:
        raise ValueError()

    reviews = read_jsonl(file_path)   
    # ps = nltk.stem.porter.PorterStemmer() # stemming e.g. ps.stem('words')
    
    parsed_reviews = []
    cnt_invalid = 0
    cnt_empty = 0
    for review in tqdm(reviews, desc="Parsing reviews", ncols=0):
        obj = { 
            "entity_id": review['entity_id'],
            "review_id": review['review_id'],
        }
        # Find all positions of occurrence of the text '#Aspect name:' in review['response']
        positions = [m.start() for m in re.finditer(r'#Aspect name:', review['response'])]
        positions.append(len(review['response']))
        sentences = [ review['response'][positions[i]: positions[i+1]].strip() for i in range(len(positions)-1)]
        sentence_data = []
        cnt_invalid_sent = 0
        for sentence in sentences:
            metadata = parse_function(sentence)
            if metadata:
                sentence_data.append(metadata)
            else:
                cnt_invalid_sent += 1
                # print(review['response'])
                # if "Aspect" in sentence and "Entity" in sentence \
                #     and "Opinion" in sentence and "Description" in sentence:
                #     print(f"Enity ID: {review['entity_id']}, Review ID: {review['review_id']}:\n{sentence}")
        obj['data'] = sentence_data
        # if len(sentence_data) == 0:
        #     sentences = [r.strip() for r in review['response'].split("\n") if r.strip().startswith('#')]
        #     sentence_data = []
        #     cnt_invalid_sent = 0
        #     for sentence in sentences:
        #         metadata = parse_function(sentence)
        #         if metadata:
        #             sentence_data.append(metadata)
        #         else:
        #             cnt_invalid_sent += 1
        #             # print(review['response'])
        #             if "Aspect" in sentence and "Feature" in sentence \
        #                 and "Opinion" in sentence and "Description" in sentence:
        #                 print(f"Enity ID: {review['entity_id']}, Review ID: {review['review_id']}:\n{sentence}")
        #     obj['data'] = sentence_data
        cnt_invalid += cnt_invalid_sent
        if len(obj['data']) == 0:
            cnt_empty += 1
            # if "Aspect" in review['response'] and "Entity" in review['response'] \
            #     and "Opinion" in review['response'] and "Description" in review['response']:
            #     print(f"Enity ID: {review['entity_id']}, Review ID: {review['review_id']}:\n{review['response']}")
        else:
            parsed_reviews.append(obj)
    logger.info(f"Discarded {cnt_invalid} invalid sentences")
    logger.info(f"Found {cnt_empty} parsed reviews with no valid sentences")
    return parsed_reviews
        

def main():
    parser = argparse.ArgumentParser(description="Updating Config settings.")
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="amasum")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    # setup logger
    global logger
    logger = setup_logger(file=args.log_file, level=args.log_level)

    parsed_reviews = parse_extraction_output(args.input_path, args.dataset)
    write_jsonl(parsed_reviews, args.output_path)

if __name__ == "__main__":
    main()
