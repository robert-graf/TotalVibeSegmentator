import argparse
import sys
import types
from enum import Enum
from inspect import Parameter
from typing import get_args, get_origin


def translation_enum_to_str(enum: type[Enum]) -> list[str]:
    return [v.name for v in enum]


def class_to_str(s):
    if issubclass(s, Enum):
        return str([f for f in dir(s) if not f.startswith("__")])
    return str(s).replace("<class '", "").replace("'>", "")


def enum_to_str(i):
    if isinstance(i, Enum):
        return i.name
    return i


def len_checker(s, num_elements, org_annotation, can_be_none, name):
    if can_be_none and s is None:
        return s
    if str(s) == "<factory>":
        return s
    if len(s) == num_elements:
        return s
    raise argparse.ArgumentTypeError(
        f"Tuple named '{name}' must must have length of {num_elements} by the definition: {org_annotation};\n A tuple has a fixed lenght, except when the definition ends with ...; Like: tuple[int,...]"
    )


# Cast list to Set/Tuple:
def cast_if_list_to(type_, val, parameter: types.GenericAlias):
    if get_origin(parameter) == type_:
        return type_(val) if isinstance(val, list) else val
    return val


def cast_if_enum(val, parameter: types.GenericAlias, enum: None | type):
    # Cast Str to Enum
    if enum is None:
        return val
    if issubclass(enum, Enum) and isinstance(val, str):
        try:
            return parameter[val]
        except KeyError:
            print(f"Enum {(enum)} has no {val}")
            sys.exit(1)
    return val


def extract_sub_annotation(annotation):
    annotations = []
    had_ellipsis = False
    has_optional = False
    for i in get_args(annotation):
        if i == types.NoneType:
            has_optional = True
        elif i == Ellipsis:
            had_ellipsis = True
        else:
            annotations.append(i)
            annotation = i
    return annotation, annotations, had_ellipsis, has_optional


def _cast_all(val, annotation: types.GenericAlias, enum):  # -> tuple[Any, ...] | set[Any] | Any | list[Any]:
    if get_origin(annotation) == types.UnionType:
        annotation = extract_sub_annotation(annotation)[0]

    if isinstance(val, list):
        ann = extract_sub_annotation(annotation)[0]
        val = [_cast_all(v, ann, enum) for v in val]
    val = cast_if_enum(val, annotation, enum)
    val = cast_if_list_to(set, val, annotation)
    val = cast_if_list_to(tuple, val, annotation)
    return val


def cast_all(val, parameter: Parameter, enum):
    if str(val) == "<factory>":
        return val
    val_out = _cast_all(val, parameter.annotation, enum)
    return val_out
