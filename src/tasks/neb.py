from src.tasks.base import Task, task_manager
import src.components as components
import src.criteria as criteria
import src.utils as utils

def neb_1(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.EmptyComponent(),
        prompt_wrapper,
        client_chat,
        components.RegexExtractor(pattern=kwargs.get("pattern")),
        components.INCAROverride(config=kwargs.get("config")),
        components.WriteFile(path=kwargs.get("filename")),
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
    ]
    return pipeline

def neb_2(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        components.GetStdout(),
        criteria.Error("neb calc_error", kwargs.get("gt")),
        criteria.NoExceptionPass("neb calc_pass")
    ]
    return pipeline

def neb_3(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.EmptyComponent(),
        prompt_wrapper,
        client_chat,
        components.RegexExtractor(pattern=kwargs.get("pattern")),
        #components.INCAROverride(config=kwargs.get("config")),
        #components.WriteFile(path=kwargs.get("filename")),
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
    ]
    return pipeline

@task_manager.register("neb")
class NEBTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipelines = [
            neb_1,
            neb_1,
            neb_3,
            neb_1,
            neb_2,
        ]