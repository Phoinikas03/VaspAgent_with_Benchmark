from src.tasks.base import task_manager
import glob
import os
import importlib

# get all .py files
current_dir = os.path.dirname(__file__)
py_files = glob.glob(os.path.join(current_dir, "*.py"))

# import all modules
for py_file in py_files:
    module_name = os.path.basename(py_file)[:-3]  # 移除 .py 后缀
    if module_name not in ['__init__', 'base']:
        importlib.import_module(f"src.tasks.{module_name}")