from src.tasks.base import Task, task_manager
import src.components as components
import src.criteria as criteria
import src.utils as utils


def bandgap_calculation_1(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.EmptyComponent(),
        prompt_wrapper,
        client_chat,
        components.RegexExtractor(pattern=kwargs.get("pattern")),
        components.WriteFile(path=kwargs.get("filename")),
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        # utils.flatten_prompt
    ]
    return pipeline

def bandgap_calculation_2(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        components.EmptyComponent()
    ]
    return pipeline

def bandgap_calculation_3(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        components.GetStdout(),
        criteria.Error("homo calc_error", kwargs.get("gt")),
        criteria.NoExceptionPass("homo calc_pass")
    ]
    return pipeline

def bandgap_calculation_4(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        components.GetStdout(),
        criteria.Error("lumo calc_error", kwargs.get("gt")),
        criteria.NoExceptionPass("lumo calc_pass")
    ]
    return pipeline


@task_manager.register("relax")
class RelaxTask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipelines = [
            bandgap_calculation_1,
            bandgap_calculation_2,
            bandgap_calculation_2,
        ]