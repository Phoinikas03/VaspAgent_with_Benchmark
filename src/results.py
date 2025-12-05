from typing import Dict
from src.criteria import Criterion

class ResultManager():
    def __init__(self):
        self.results: Dict = {}

    def add_result(self, result: Criterion):
        title = result.title
        keys = result.metrics.keys()
        if len(keys) == 1:
            print(f"Result {title} has only one key {keys}. We will not record this result.")
        for key, value in result.metrics.items():
            if key == "n": continue
            if value is None:
                print(f"Result {title} has None value for {key}. We will not record this result.")
        if title not in self.results:
            self.results[title] = result.metrics
        else:
            for key, value in result.metrics.items():
                if key == "n": continue
                self.results[title][key] *= self.results[title]["n"]
                self.results[title][key] += value
                self.results[title][key] /= (self.results[title]["n"] + result.metrics["n"])
            self.results[title]["n"] += result.metrics["n"]

    def __repr__(self) -> str:
        header = f"{self.__class__.__name__}"
        results = ""
        for key, value in self.results.items():
            results += f"{key}: {value}\n"
        
        # Get the maximum width for the separator line
        max_width = max(len(header), max((len(line) for line in results.splitlines()), default=0))
        separator = "-" * max_width
        
        return f"{header}\n{separator}\n{results}"

    def reset(self):
        self.results = {}