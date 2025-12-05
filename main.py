import argparse
from src.tasks import task_manager
from src.agents import agent_manager
from src.llm import ChatEngine
import src.utils as utils
from itertools import product
import threading
import os, json
import src.prompts as prompts

def main(task_name, model, directory_or_folders, num_threads, prompt_keys):
    title = f"{task_name}_{model}"
    log_root = utils.init_logging(title)
    system_prompt = prompts.DEFAULT_SYSTEM_PROMPT + prompts.VASP_PROMPT + prompts.FORMAT_PROMPT
    pattern = r"```vasp(.*?)```"

    if isinstance(directory_or_folders, str):
        tasks = task_manager.from_directory(task_name, directory_or_folders, log_root=log_root, prompt_keys=prompt_keys, pattern=pattern)
    elif isinstance(directory_or_folders, list):
        tasks = task_manager.from_folders(task_name, directory_or_folders, log_root=log_root, prompt_keys=prompt_keys, pattern=pattern)
    total_num = len(tasks)
    clients = ChatEngine.spawn(total_num, model=model, system_prompt=system_prompt)
    assert num_threads <= total_num

    threads = []
    results = []

    task_groups = [[] for _ in range(num_threads)]
    client_groups = [[] for _ in range(num_threads)]
    for i in range(total_num):
        client_groups[i % num_threads].append(clients[i])
        task_groups[i % num_threads].append(tasks[i])

    for i in range(num_threads):
        t = threading.Thread(target=worker, args=(i, client_groups[i], task_groups[i], results))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()

    print("All threads finished")

def worker(index, clients, tasks, results):
    print(f"Worker {index} started")
    # modify some environment variables
    # os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
    for client, task in zip(clients, tasks):
        result = task.run(client)
        results.append(result)
    print(f"Worker {index} finished")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="")

    args = parser.parse_args()
    if args.config:
        with open(args.config, "r") as f:
            config = json.load(f) 
        task_name = config["task_name"]
        directory_or_folders = config["directory_or_folders"]
        num_threads = config["num_threads"]
        model = config["model"]
        prompt_keys = config["prompt_keys"]
        main(task_name, model, directory_or_folders, num_threads, prompt_keys)
    else:
        print("Please specify a config file")