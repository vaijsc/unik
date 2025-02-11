import json

def generate_json_schema(data):
    """
    Generate a JSON schema from the provided JSON object.
    """
    if isinstance(data, dict):
        return {
            "type": "object",
            "properties": {
                key: generate_json_schema(value) for key, value in data.items()
            },
            "required": list(data.keys())  # Assume all keys are required
        }
    elif isinstance(data, list):
        if len(data) > 0:
            return {
                "type": "array",
                "items": generate_json_schema(data[0])
            }
        else:
            return {"type": "array", "items": {}}
    elif isinstance(data, str):
        return {"type": "string"}
    elif isinstance(data, int):
        return {"type": "integer"}
    elif isinstance(data, float):
        return {"type": "number"}
    elif isinstance(data, bool):
        return {"type": "boolean"}
    elif data is None:
        return {"type": "null"}
    else:
        return {"type": "unknown"}

def generate_json_structure(json_object):
    """
    Generate a JSON schema from the provided JSON object.
    """
    if isinstance(json_object, dict):
        return {
            key: generate_json_structure(value) for key, value in json_object.items()
        }
    elif isinstance(json_object, list):
        if len(json_object) > 0:
            return [generate_json_structure(json_object[0])]
        else:
            return [] 
    elif isinstance(json_object, str):
        return "str" 
    elif isinstance(json_object, int):
        return "int" 
    elif isinstance(json_object, float):
        return "float" 
    elif isinstance(json_object, bool):
        return "bool" 
    elif json_object is None:
        return "null"
    else:
        return "unknown" 

if __name__ == "__main__":
    from src.utils import read_jsonl
    import sys
    input_path = sys.argv[1] if len(sys.argv) > 1 else 'data/raw/amasum/full/dev.jsonl'
    json_obj = read_jsonl(input_path, 1)[0]
    json_schema = generate_json_structure(json_obj)
    print(json.dumps(json_schema, indent=4))
