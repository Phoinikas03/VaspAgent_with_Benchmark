import json

# 需要设置为 True 的参数列表（自动添加 .html 后缀）
target_params = {
    "SYSTEM", "PREC", "ENCUT", "EDIFF", "EDIFFG",
    "IBRION", "ISIF", "NSW", "POTIM", 
    "ISMEAR", "SIGMA", "LORBIT", 
    "ALGO", "LREAL", "LWAVE", "LCHARG"
}

# 1. 读取原始文件
with open("config.json", "r") as f:
    data = json.load(f)

# 2. 处理每个参数
processed = {
    key: key.replace(".html", "") in target_params
    for key in data
}

# 3. 保存结果
with open("processed_config.json", "w") as f:
    json.dump(processed, f, indent=2)

print("处理完成，结果已保存到 processed_config.json")