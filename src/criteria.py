from typing import Dict, List
import src.utils as utils
from collections import defaultdict

# grouped these criteria by title with result_manager

class Criterion():
    def __init__(self, title):
        self.title: str = title
        self.metrics: Dict = defaultdict(None)
        self.metrics["n"] = 1 # batch size

    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.metrics})"

class Accuracy(Criterion):
    """Accuracy for values
    could be 1234
    could be abcd
    """
    def __init__(self, title, gt):
        super().__init__(title)
        self.gt = utils.string_to_numbser(gt)

    def __call__(self, content):
        content = utils.type_match(content, self.gt)
        self.metrics["accuracy"] = content == self.gt
        # self.metrics["precision"] = content == self.gt
        # self.metrics["recall"] = content == self.gt
        # self.metrics["f1"] = content == self.gt
        return self.metrics

class NoExceptionPass(Criterion):
    """No Exception Pass for True or False
    """
    def __init__(self, title, exception=BaseException):
        super().__init__(title)
        self.exception = exception

    def __call__(self, content):
        if isinstance(content, self.exception):
            self.metrics["pass"] = False
        else:
            self.metrics["pass"] = True
        return self.metrics

class Error(Criterion):
    """Error for values
    """
    eps = 1e-12

    def __init__(self, title, gt):
        super().__init__(title)
        self.gt = utils.string_to_numbser(gt)

    def __call__(self, content):
        content = utils.type_match(content, self.gt)
        self.metrics["abs_error"] = abs(self.gt - content)
        self.metrics["rel_error"] = abs(self.gt - content) / (self.gt + self.eps)
        print(self.gt, content)
        print(abs(self.gt - content))
        return self.metrics