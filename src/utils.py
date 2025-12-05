import sys
import os, logging
from datetime import datetime
# import numpy as np
# import torch

class Registry():
    def __init__(self):
        self._registry = {}

    def register(self, name):
        def decorator(class_or_func):
            self._registry[name] = class_or_func
            return class_or_func
        return decorator

    def build(self, name, **kwargs):
        if name not in self._registry:
            raise ValueError(f"{self.__class__.__name__}: {name} not found")
        return self._registry[name](**kwargs)

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__}"
        _list = ""
        for index, (name, item) in enumerate(self._registry.items()):
            _list += f"{index+1}. {name}: {item}\n"
        max_width = max(len(header), max((len(line) for line in _list.splitlines()), default=0))
        separator = "-" * max_width
        return f"{header}\n{separator}\n{_list}"

def config_logging(log_file="main.log"):
    date_format = '%Y-%m-%d %H:%M:%S'
    log_format = '[%(asctime)s] [%(levelname)s]:\n%(message)s'
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))

    # Set up the FileHandler for logging to a file
    file_handler = logging.FileHandler(log_file, mode='a')
    file_handler.setFormatter(logging.Formatter(log_format, datefmt=date_format))
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])

def init_logging(title="test", root="runs/"):
    date = datetime.now().strftime("%Y_%m_%d")
    time = datetime.now().strftime("%H_%M_%S")
    log_dir = os.path.join(root, date, f"{title}-{time}")
    os.makedirs(log_dir, exist_ok=True)
    config_logging(os.path.join(log_dir, "main.log"))
    return log_dir

def flatten_dict_or_list(prompt_data) -> str:
    """Flatten prompts to a string.
    """
    prompt = ""
    if isinstance(prompt_data, dict):
        for key, value in prompt_data.items():
            prompt += f"{key}:\n" + flatten_dict_or_list(value)
    elif isinstance(prompt_data, list):
        for index, item in enumerate(prompt_data):
            prompt += f"{index+1}. " + flatten_dict_or_list(item)
    else:
        prompt += f"{prompt_data}\n"
    return prompt

def string_to_numbser(item):
    if isinstance(item, str):
        if item.isdigit(): 
            item = int(item)
        elif item.replace(".", "", 1).isdigit() and item.count(".") < 2:
            item = float(item)
    return item

def type_match(item, ref):
    """convert item to type of ref
    consider basic types, like int, float, str, list, tuple, dict, set, bool, None
    consider complex types, like np.array, torch.Tensor
    TODO: convert string number to number? ref number must not be string now!
    """
    if type(item) != type(ref):
        # Handle basic types
        if isinstance(ref, (int, float)):
            return float(item)
        elif isinstance(ref, (str, bool)):
            return type(ref)(item)
        # Handle container types
        elif isinstance(ref, (list, tuple)):
            return type(ref)(type_match(x, ref[0]) if ref else x for x in item)
        elif isinstance(ref, dict):
            if not ref:
                return type(ref)(item)
            ref_key, ref_val = next(iter(ref.items()))
            return type(ref)({type_match(k, ref_key): type_match(v, ref_val) for k, v in item.items()})
        elif isinstance(ref, set):
            return type(ref)(type_match(x, next(iter(ref))) if ref else x for x in item)
        # # Handle numpy arrays
        # elif isinstance(ref, np.ndarray):
        #     return np.array(item, dtype=ref.dtype)
        # # Handle PyTorch tensors
        # elif isinstance(ref, torch.Tensor):
        #     tmp_ref = ref.numpy()
        #     tmp_item = np.array(item, dtype=tmp_ref.dtype)
        #     return torch.from_numpy(tmp_item).to(dtype=ref.dtype, device=ref.device)
        # Handle None type
        elif ref is None:
            return None
        else:
            raise TypeError(f"Unsupported type conversion from {type(item)} to {type(ref)}")
    return item