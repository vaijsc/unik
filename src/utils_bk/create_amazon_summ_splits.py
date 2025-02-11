import csv 
def read_data_csv(path):
    with open(path) as f:
        reader = csv.DictReader(f, delimiter='\t')
        entities = []
        for row in reader:
            entities.append(row)
    return entities

if __name__ == "__main__":
    dev_file_path = 'data/amazon/dev.csv'
    test_file_path = 'data/amazon/test.csv'
    split_file_path = 'data/amazon/amazon_summ_splits.txt'
    dev_entities = read_data_csv(dev_file_path)
    test_entities = read_data_csv(test_file_path)
    dev_entity_ids = [e['prod_id'] for e in dev_entities]
    test_entity_ids = [e['prod_id'] for e in test_entities]
    print('len dev', len(dev_entity_ids))
    print('len test', len(test_entity_ids))
    print('len all (unique)', len(set(dev_entity_ids + test_entity_ids)))

    # Save to file
    splits = []
    for e in dev_entity_ids:
        splits.append((e, 'dev'))
    for e in test_entity_ids:
        splits.append((e, 'test'))
    with open(split_file_path, 'w') as f:
        for e, t in splits:
            f.write(f"{e}\t{t}\n") 
    print(f"Saved to {split_file_path}")
