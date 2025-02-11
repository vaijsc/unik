import argparse
from collections import defaultdict, OrderedDict
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from neo4j import GraphDatabase
from tqdm import tqdm
from src.utils import read_jsonl, setup_logger
from src.config import cfg

logger = None
DB = "neo4j"


class Editor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.stopwords = set(stopwords.words("english"))

    def edit(self, text):
        words = word_tokenize(text)
        words = [w for w in words if w.lower() not in self.stopwords]
        stemmed_words = [self.stemmer.stem(word) for word in words]
        stemmed_text = " ".join(stemmed_words)
        return stemmed_text


def get_driver(uri, username, password):
    return GraphDatabase.driver(uri, auth=(username, password))


def push_data(entity_id, entity_data, editor, driver, exp_name, dataset_name):
    # Create entity and aspects
    entity_params = {}
    if dataset_name == "space":
        entity_params = {
            "entity_id": entity_id,
            "exp_name": exp_name,
            "aspects": [
                {"id": f"{entity_id}_general", "name": "general"},
                {"id": f"{entity_id}_rooms", "name": "rooms"},
                {"id": f"{entity_id}_location", "name": "location"},
                {"id": f"{entity_id}_service", "name": "service"},
                {"id": f"{entity_id}_cleanliness", "name": "cleanliness"},
                {"id": f"{entity_id}_building", "name": "building"},
                {"id": f"{entity_id}_food", "name": "food"},
            ],
        }
    elif dataset_name == "amasum":
        entity_params = {
            "entity_id": entity_id,
            "exp_name": exp_name,
            "aspects": [
                {"id": f"{entity_id}_general", "name": "general"},
            ],
        }
    else:
        raise ValueError()

    # add entity_data
    graph_data = []
    review_count_by_feature = defaultdict(lambda: 0)
    for _, review_data in entity_data.items():
        reviewed_features = set()
        for sentence_data in review_data:
            aspect = sentence_data["aspect"]
            feature = editor.edit(sentence_data["feature"])
            # opinions = list(set([editor.edit(o) for o in sent['opinions']]))
            opinions = sentence_data["opinions"]
            opinions = [o for o in opinions if o != ""]
            description = sentence_data["description"]

            for opinion in opinions:
                graph_data.append(
                    {
                        "entity_id": entity_id,
                        "aspect": aspect,
                        "feature": feature,
                        "opinion": opinion,
                        "description": description,
                    }
                )

            reviewed_features.add(feature)
        for feature in reviewed_features:
            review_count_by_feature[feature] += 1

    # update review count on feature
    feature_count_list = [{"feature": k, "count": v} for k, v in review_count_by_feature.items()]

    # command
    cmd_create_entity = """
MERGE (e:Entity {entity_id: $entity_id, exp_name: $exp_name})
WITH e
UNWIND $aspects as aspect
MERGE (e)-[:HAS_ASPECT]-(a:Aspect {entity_id: $entity_id, name: aspect.name, exp_name: $exp_name})
"""
#     cmd_create_entity = """
# CALL apoc.periodic.iterate(
#     "MERGE (e:Entity {entity_id: $entity_id, exp_name: $exp_name}) RETURN e",
#     "UNWIND $aspects as aspect
#     MERGE (e)-[:HAS_ASPECT]->(a:Aspect {entity_id: $entity_id, name: aspect.name, exp_name: $exp_name})",
#     {batchSize: 1000, params: {aspects: $aspects, entity_id: $entity_id, exp_name: $exp_name}}
# )
#     """
    cmd_create_other_data = """
UNWIND $graph_data as fe
MATCH (e:Entity {entity_id: fe.entity_id, exp_name: $exp_name})-[]-(a:Aspect {name: fe.aspect})
MERGE (a)-[:HAS_FEATURE]->(f:Feature {entity_id: fe.entity_id, name: fe.feature, exp_name: $exp_name})
MERGE (f)-[r:HAS_OPINION]->(o:Opinion {entity_id: fe.entity_id, name: fe.opinion, exp_name: $exp_name})
ON CREATE SET r.entity_id = fe.entity_id, r.exp_name = $exp_name
WITH r, fe.aspect as aspect, fe.description as description
CALL apoc.create.setRelProperty(r, aspect, coalesce(r[aspect], []) + description) YIELD rel
RETURN rel
"""
#     cmd_create_other_data = """
# CALL apoc.periodic.iterate(
#     "UNWIND $graph_data as fe RETURN fe",
#     "MATCH (e:Entity {entity_id: fe.entity_id})-[:HAS_ASPECT]->(a:Aspect {name: fe.aspect, exp_name: $exp_name})
#     MERGE (a)-[:HAS_FEATURE]->(f:Feature {name: fe.feature, exp_name: $exp_name})
#     MERGE (f)-[:HAS_OPINION]->(o:Opinion {name: fe.opinion, exp_name: $exp_name})
#     ON CREATE SET o.entity_id = fe.entity_id,
#                     f.entity_id = fe.entity_id,
#                     r.entity_id = fe.entity_id,
#                     r.exp_name = $exp_name
#     WITH r, fe.aspect as aspect, fe.description as description
#     CALL apoc.create.setRelProperty(r, aspect, coalesce(r[aspect], []) + description) YIELD rel",
#     {batchSize: 1000, params: {graph_data: $graph_data, exp_name: $exp_name}}
# )
#     """
    cmd_set_feature_count = """
UNWIND $feature_count as fc
MATCH (e:Entity {entity_id: $entity_id, exp_name: $exp_name})-[]-(:Aspect)-[]-(f:Feature {name: fc.feature})
SET f.count = fc.count
"""
#     cmd_set_feature_count = """
# CALL apoc.periodic.iterate(
#     "UNWIND $feature_count as fc RETURN fc",
#     "MATCH (e:Entity {entity_id: $entity_id}, exp_name: $exp_name)-[:HAS_ASPECT]-(a:Aspect)-[:HAS_FEATURE]-(f:Feature {name: fc.feature, exp_name: $exp_name})
#     SET f.count = fc.count",
#     {batchSize: 1000, params: {feature_count: $feature_count, entity_id: $entity_id, exp_name: $exp_name}}
# )
# """
    with driver.session(database=DB) as session:
        try:
            with session.begin_transaction() as tx:
                tx.run(cmd_create_entity, **entity_params)
                tx.run(cmd_create_other_data, **{"graph_data": graph_data, "exp_name": exp_name})
                tx.run(
                    cmd_set_feature_count,
                    **{"feature_count": feature_count_list, "entity_id": entity_id, "exp_name": exp_name},
                )

                # Commit the transaction if everything is successful
                tx.commit()
        except Exception as e:
            logger.error(f"Transaction failed: {e}") 
            logger.error(f"Failed entity_id: {entity_id}")
            raise e  # Optionally re-raise the exception for further handling


def get_processed_entities(exp_name, driver):
    """
    Query the database to get the list of entity_ids that are already in the database
    """
    cmd = """
MATCH (e:Entity {exp_name: $exp_name})
RETURN e.entity_id as entity_id
"""
    result, _, _ = driver.execute_query(cmd, parameters_={"exp_name": exp_name}, database_=DB)
    return [r['entity_id'] for r in result]


def main():
    parser = argparse.ArgumentParser(description="Updating Config settings.")
    parser.add_argument(
        "--log_level", type=str, default="INFO", help="Set the logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)"
    )
    parser.add_argument("--log_file", type=str, default=None)
    parser.add_argument("--input_path", type=str, default=None)
    parser.add_argument("--dataset", type=str, default="amasum")
    parser.add_argument("--exp_name", type=str, default="amasum_mistral")
    args = parser.parse_args()
    cfg.update(args)

    # setup logger
    global logger
    logger = setup_logger(file=args.log_file, level=args.log_level)

    # logger.info(f"Config:\n{cfg}")

    # Parse the input data
    parsed_data = read_jsonl(cfg.CONF["input_path"])
    entities = OrderedDict()
    for d in parsed_data:
        entity_id = d["entity_id"]
        review_id = d["review_id"]
        if entity_id not in entities:
            entities[entity_id] = {}
        if review_id not in entities[entity_id]:
            entities[entity_id][review_id] = []
        entities[entity_id][review_id] += d["data"]

    # Connect to the database
    driver = get_driver(**cfg.DB_CONF["neo4j"])

    # Get list of entity_ids that are already in the database
    processed_entities = get_processed_entities(cfg.CONF["exp_name"], driver)
    # print(processed_entities)
    logger.info(f"Found {len(processed_entities)} entities in the database.")

    # Remove already processed entities
    entities = {k: v for k, v in entities.items() if k not in processed_entities}

    editor = Editor()
    p_bar = tqdm(total=len(entities), desc="Processing entities", ncols=0)
    for entity_id, entity_data in entities.items():
        # logger.info(f"Processing entity: {entity_id}")
        push_data(
            entity_id, entity_data, editor, driver, exp_name=cfg.CONF["exp_name"], dataset_name=cfg.CONF["dataset"]
        )
        p_bar.update(1)
    p_bar.close()
    driver.close()  # Dont forget to close driver


if __name__ == "__main__":
    main()
