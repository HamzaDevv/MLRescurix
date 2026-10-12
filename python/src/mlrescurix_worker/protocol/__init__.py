"""Bounded JSON frames validated against the packaged versioned schema.

This is a strict subset validator for our schema vocabulary, not a general
JSON Schema library. Unknown schema keywords fail closed.
"""
import json
import re
from pathlib import Path

MAX_FRAME = 65_536
SCHEMA = json.loads(Path(__file__).with_name("frame.schema.json").read_text())

class ProtocolError(ValueError):
    pass


def _validate(value, schema, root):
    allowed = {"$schema", "$id", "$defs", "$ref", "type", "const", "enum", "oneOf",
               "properties", "required", "additionalProperties", "items", "maxItems",
               "minItems", "minimum", "maximum", "minLength", "maxLength", "pattern",
               "maxProperties", "minProperties", "maxUtf8Bytes", "description"}
    if set(schema) - allowed:
        raise ProtocolError("unsupported schema keyword")
    if "$ref" in schema:
        node = root
        for piece in schema["$ref"].removeprefix("#/").split("/"):
            node = node[piece]
        _validate(value, node, root)
        return
    if "oneOf" in schema:
        successes = 0
        for alternative in schema["oneOf"]:
            try:
                _validate(value, alternative, root)
                successes += 1
            except ProtocolError:
                pass
        if successes != 1:
            raise ProtocolError("oneOf mismatch")
    if "const" in schema and (type(value) is not type(schema["const"]) or value != schema["const"]):
        raise ProtocolError("constant mismatch")
    if "enum" in schema and value not in schema["enum"]:
        raise ProtocolError("enum mismatch")
    types = {"object": dict, "array": list, "string": str, "integer": int, "null": type(None), "boolean": bool}
    if "type" in schema and type(value) is not types[schema["type"]]:
        raise ProtocolError("type mismatch")
    if type(value) is dict:
        if len(value) < schema.get("minProperties", 0) or len(value) > schema.get("maxProperties", len(value)):
            raise ProtocolError("object size")
        if not set(schema.get("required", [])) <= value.keys():
            raise ProtocolError("missing property")
        properties = schema.get("properties", {})
        for key, item in value.items():
            if key in properties:
                _validate(item, properties[key], root)
            elif schema.get("additionalProperties") is False:
                raise ProtocolError("unknown property")
            elif isinstance(schema.get("additionalProperties"), dict):
                _validate(item, schema["additionalProperties"], root)
    if type(value) is list:
        if len(value) < schema.get("minItems", 0) or len(value) > schema.get("maxItems", len(value)):
            raise ProtocolError("array size")
        for item in value:
            _validate(item, schema["items"], root)
    if type(value) is str:
        if len(value) < schema.get("minLength", 0) or len(value) > schema.get("maxLength", len(value)):
            raise ProtocolError("string length")
        if len(value.encode("utf-8")) > schema.get("maxUtf8Bytes", 2**63):
            raise ProtocolError("string byte bound")
        if "pattern" in schema and re.fullmatch(schema["pattern"], value, flags=re.ASCII) is None:
            raise ProtocolError("string pattern")
    if type(value) is int and (value < schema.get("minimum", value) or value > schema.get("maximum", value)):
        raise ProtocolError("integer range")


def decode_frame(data: bytes) -> dict:
    if len(data) > MAX_FRAME:
        raise ProtocolError("frame too large")
    try:
        value = json.loads(data.decode("utf-8"), parse_constant=lambda _: (_ for _ in ()).throw(ProtocolError("nonfinite JSON")))
        _validate(value, SCHEMA, SCHEMA)
        return value
    except (ValueError, UnicodeError, RecursionError) as exc:
        raise ProtocolError(str(exc)) from exc


def encode_frame(value: dict) -> bytes:
    data = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")
    decode_frame(data)
    return data
