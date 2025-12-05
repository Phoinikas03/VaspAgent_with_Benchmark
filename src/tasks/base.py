from src.criteria import Criterion
import os, shutil, logging
from src.results import ResultManager
from src.utils import flatten_dict_or_list, Registry
import json
import src.components as components
import src.criteria as criteria

logger = logging.getLogger(__name__)

class Task():
    """Load dataset, run pipelines, save results
    each pipeline is a list of components, with at most one llm call
    task is a list of pipelines
    """
    pipelines: list # list of pipelines
    results: ResultManager # save result_manager

    def __init__(self, folder, data_json="data.json", log_root="runs/", pattern=r"(.*)", 
                prompt_keys=[], gt_key="gt", pattern_key="pattern", filename_key="filename", command_key="command"):
        self.folder = folder
        self.directory = os.path.basename(os.path.normpath(folder))
        self.data_json = data_json

        self.log_root = log_root
        self.pattern = pattern

        self.prompt_keys = prompt_keys
        self.gt_key = gt_key
        self.pattern_key = pattern_key
        self.filename_key = filename_key
        self.command_key = command_key

        os.makedirs(self.log_root, exist_ok=True)

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__}({self.directory})"
        pipeline_list = ""
        for index, pipeline in enumerate(self.pipelines):
            pipeline_list += f"{index + 1}. {pipeline}\n"

        max_width = max(len(header), max((len(line) for line in pipeline_list.splitlines()), default=0))
        separator = "-" * max_width
        return f"{header}\n{separator}\n{pipeline_list}"

    def _validate(self):
        if (not os.path.isdir(self.folder)) or\
            (not os.path.exists(os.path.join(self.folder, self.data_json))): 
            logger.warning(f"Invalid directory: {self.folder} without {self.data_json}")
            return False
        return True

    def _create_log_dir(self):
        self.log_dir = os.path.join(self.log_root, self.directory)
        os.makedirs(self.log_dir, exist_ok=True)
        return self.log_dir

    def run(self, client):
        result_manager = ResultManager()

        # handle directory data
        if not self._validate(): return result_manager # pass if invalid directory or data.json

        logger.info(f"Running task:\n{self.directory}")
        self._create_log_dir()
        client.reset()
        client_chat = client.chat

        # load new data and reset context
        content = ""
        # create new directory and work here
        cwd = self.log_dir
        shutil.copytree(self.folder, cwd, dirs_exist_ok=True)
        data_json = os.path.join(cwd, self.data_json)

        with open(data_json, "r") as f:
            data = json.load(f)
        print(data)
        print(self.pipelines)
        assert len(data) == len(self.pipelines), f"data does not match pipeline length"
        for step, pipeline in zip(data, self.pipelines):
            # llm can accept context more than just from data.json prompt
            # use EmptyComponent if you want to block the context
            prompt_data = {k: step.pop(k, None) for k in self.prompt_keys}
            prompt = flatten_dict_or_list(prompt_data)
            
            # WARNING: previous context is concatenated to current prompt
            def prompt_wrapper(x):
                placeholder = ""
                if x:
                    placeholder = "Previous context is: \n" + x + "\n\n Current prompt is: \n"
                return placeholder + prompt

            gt = step.pop(self.gt_key, None)
            pattern = step.pop(self.pattern_key, self.pattern)
            filename = step.pop(self.filename_key, "")
            if filename: filename = os.path.join(cwd, filename) # relative to cwd
            command = step.pop(self.command_key, "")
            config = step.pop("config", {})

            for component in pipeline(client_chat, prompt_wrapper, 
                                    gt=gt, pattern=pattern, filename=filename, 
                                    command=command, config=config, cwd=cwd, **step):
                try:
                    content_display = flatten_dict_or_list(content) if isinstance(content, dict) or isinstance(content, list) else str(content)
                    logger.info(f"{content_display} -> {component}")
                    content = component(content)
                except Exception as e:
                    logger.info(f"Exception occurs when running {component}:\n{e}")
                    content = e
                if isinstance(component, Criterion):
                    result_manager.add_result(component)
        
        # TODO: double check history messages
        logger.info("Client history messages:\n" + str(client.messages))

        logger.info(f"{result_manager}")
        with open(os.path.join(self.log_dir, "result.txt"), "w") as f:
            f.write(str(result_manager))
        return result_manager

    @classmethod
    def from_directory(cls, dirpath, **kwargs):
        """Load task from directory"""
        folders = []
        for directory in os.listdir(dirpath):
            folder = os.path.join(dirpath, directory)
            folders.append(folder)
        tasks = cls.from_folders(folders, **kwargs)
    
        return tasks
    
    @classmethod
    def from_folders(cls, folders, **kwargs):
        """Load task from list of folders"""
        tasks = []
        for folder in folders:
            task = cls(folder, **kwargs)
            if task._validate(): 
                tasks.append(task)
            else:
                logger.warning(f"Skipping directory: {folder}")

        logger.info(f"Loaded {len(tasks)} tasks from {folders}")
        return tasks

class TaskManager(Registry):
    def from_directory(self, name, dirpath, **kwargs):
        """Load task from directory"""
        if name not in self._registry:
            raise ValueError(f"{self.__class__.__name__}: {name} not found")
        return self._registry[name].from_directory(dirpath, **kwargs)

    def from_folders(self, name, folders, **kwargs):
        """Load task from list of folders"""
        if name not in self._registry:
            raise ValueError(f"{self.__class__.__name__}: {name} not found")
        return self._registry[name].from_folders(folders, **kwargs)

# used to register tasks
task_manager = TaskManager()

##### Basic Test Tasks #####

"""one pipeline for at most one llm call
def pipeline(client_chat, prompt_wrapper, **kwargs):
    ...
"""

def arithmetic_calculation(client_chat, prompt_wrapper, gt=4, pattern=r"<execute>(.*?)</execute>", **kwargs):
    pipeline = [
        components.EmptyComponent(),
        prompt_wrapper,
        client_chat,
        components.RegexExtractor(pattern=pattern),
        criteria.Error("calc_error", gt)
    ]
    return pipeline

@task_manager.register("arithmetic")
class ArithmeticTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipelines = [
            arithmetic_calculation,
            arithmetic_calculation
        ]