from src.tasks.base import Task, task_manager
import src.components as components
import src.criteria as criteria
import src.utils as utils

def absoprtion_e_1(client_chat, prompt_wrapper, **kwargs):
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

def absoprtion_e_2(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        components.EmptyComponent()
    ]
    return pipeline

def absoprtion_e_3(client_chat, prompt_wrapper, **kwargs):
    pipeline = [
        components.Command(cwd=kwargs.get("cwd"), command=kwargs.get("command")),
        components.GetStdout(),
        criteria.Error("absoprtion_e calc_error", kwargs.get("gt")),
        criteria.NoExceptionPass("absoprtion_e calc_pass")
    ]
    return pipeline

@task_manager.register("absorption_e")
class AbsorptionETask(Task):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipelines = [
            absoprtion_e_1,
            # absoprtion_e_2,
            absoprtion_e_1,
            # absoprtion_e_2,
            absoprtion_e_1,
            # absoprtion_e_2,
            absoprtion_e_3,
        ]