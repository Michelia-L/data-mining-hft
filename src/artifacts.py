"""Readable strict JSON with compact primitive arrays for generated time series."""
import json


def pretty_json(value, level=0):
    indent = '  ' * level
    child = '  ' * (level + 1)
    if isinstance(value, dict) and value:
        return '{\n' + ',\n'.join(child + json.dumps(k, ensure_ascii=False) + ': ' + pretty_json(v, level + 1)
                                  for k, v in value.items()) + '\n' + indent + '}'
    if isinstance(value, list) and any(isinstance(x, (dict, list)) for x in value):
        return '[\n' + ',\n'.join(child + pretty_json(v, level + 1) for v in value) + '\n' + indent + ']'
    return json.dumps(value, ensure_ascii=False, allow_nan=False)
