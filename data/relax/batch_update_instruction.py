#!/usr/bin/env python3
"""
批量修改 relax 目录下各材料 data.json 中的 instruction 为统一模板。
材料名从所在子目录名读取（如 Al、BN、MgO 等）。
运行方式：在 relax 目录下执行 python batch_update_instruction.py
"""

import json
from pathlib import Path

# 脚本所在目录即 relax 目录
RELAX_DIR = Path(__file__).resolve().parent

# 统一的 instruction 模板，{material} 会被替换为材料名（目录名）
INSTRUCTION_TEMPLATE = (
    "You are a materials science expert responsible for optimizing the structure of {material} "
    "based on the given initial structure to achieve the lowest total energy. "
    "The goal of the optimization is to find the most stable structure of the material under the given conditions. "
    "The paths of POSCAR, KPOINTS, POTCAR file path are 'POSCAR', 'KPOINTS', 'POTCAR'. "
    "You need to write an INCAR file for structure optimization."
)


def main():
    updated = []
    for data_path in sorted(RELAX_DIR.glob("*/data.json")):
        material = data_path.parent.name
        with open(data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list) or len(data) == 0:
            print(f"  跳过（非列表或为空）: {data_path}")
            continue
        first = data[0]
        if "instruction" not in first:
            print(f"  跳过（无 instruction）: {data_path}")
            continue
        new_instruction = INSTRUCTION_TEMPLATE.format(material=material)
        if first["instruction"] == new_instruction:
            print(f"  未改动: {data_path}")
            continue
        first["instruction"] = new_instruction
        with open(data_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        updated.append(str(data_path))
        print(f"  已更新: {data_path}")
    print(f"\n共更新 {len(updated)} 个文件。")


if __name__ == "__main__":
    main()
