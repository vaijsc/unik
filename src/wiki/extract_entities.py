import bz2
import json
from tqdm import tqdm

wikidata_path = "/lustre/scratch/client/vinai/users/hoangnv49/whoqa/tmp/wiki/wikidata-20240219-all.json.bz2"
output_path = 'entities.jsonl'
cnt = 0
batch_size = 10000

with bz2.BZ2File(wikidata_path) as fin:
    entities = []
    for line in tqdm(fin):
        line = line.decode().strip()

        if line in {"[", "]"}:
            continue
        if line.endswith(","):
            line = line[:-1]
        entity = json.loads(line)
        try:
            obj = {
                "id": entity['id'],
                "label": entity['labels']['en']['value'],
                "desc": entity['descriptions']['en']['value']
            }
            entities.append(obj)
        except Exception as e:
            continue

        if len(entities) >= batch_size:
            with open(output_path, 'a') as fout:
                for d in entities:
                    fout.write(json.dumps(d))
                    fout.write("\n")
            entities = []

        # valid_codes = ["en", "en-us", "en-au", "en-ca"]
        # # for code in valid_codes:
        # #     if code in list(entity["aliases"].keys()):
        # #         name_mapping[entity["id"]] = []
        # #         for value in entity["aliases"][code]:
        # #             name_mapping[entity["id"]].d
        # #         name_mapping[entity["id"]].append(entity["labels"][code]["value"])
        # #         break
        # flag_1, flag_2 = False, False
        #
        # name_mapping[entity["id"]] = []
        # for code in valid_codes:
        #     if flag_1 and flag_2:
        #         break
        #
        #     if code in entity["aliases"] and not flag_1:
        #         name_mapping[entity["id"]] = [value["value"] for value in entity["aliases"][code]]
        #         flag_1 = True
        #
        #     if code in entity["labels"] and not flag_2:
        #         name_mapping[entity["id"]].append(entity["labels"][code]["value"])
        #         flag_2 = True
        # if(i % 10000) == 0:
        #     print(i, "lines has gone")

# with open('/lustre/scratch/client/vinai/users/hieupq1/ExQA/wiki_dump/wikidata-id-name-map_.pkl', 'wb') as fp:
#     pickle.dump(name_mapping, fp)
    print("Done")
