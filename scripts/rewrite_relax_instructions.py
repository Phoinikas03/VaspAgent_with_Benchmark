#!/usr/bin/env python3
"""
重写 data/relax 下各材料 data.json 的 instruction 字段。
只保留指定模板，并将材料名替换为对应目录名。
"""
import json
import os

RELAX_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "data", "relax"
)

INSTRUCTION_TEMPLATE = (
    "You are a materials science expert responsible for optimizing the structure of {material} "
    "based on the given initial structure to achieve the lowest total energy. "
    "The goal of the optimization is to find the most stable structure of the material under the given conditions. "
    "The paths of POSCAR, KPOINTS, POTCAR file path are 'POSCAR', 'KPOINTS', 'POTCAR'. "
    "You need to write an INCAR file for structure optimization."
)


def main():
    for name in sorted(os.listdir(RELAX_DIR)):
        dirpath = os.path.join(RELAX_DIR, name)
        if not os.path.isdir(dirpath):
            continue
        data_path = os.path.join(dirpath, "data.json")
        if not os.path.isfile(data_path):
            print(f"Skip (no data.json): {name}")
            continue
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            print(f"Skip (invalid format): {name}")
            continue
        first = data[0]
        if not isinstance(first, dict) or "instruction" not in first:
            print(f"Skip (no instruction): {name}")
            continue
        first["instruction"] = INSTRUCTION_TEMPLATE.format(material=name)
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Updated: {name}")


if __name__ == "__main__":
    main()
