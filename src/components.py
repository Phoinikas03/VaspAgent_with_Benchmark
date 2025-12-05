import re
import subprocess
from src.config import TIMEOUT

class Component():
    def __init__(self):
        self.to_print: str = ""

    def __call__(self, content):
        return content

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.to_print})"

class EmptyComponent(Component):
    def __call__(self, content):
        return ""

class RegexExtractor(Component):
    """Extract with regular expression.
    """
    def __init__(self, pattern: str):
        self.pattern = pattern
        self.to_print = f"pattern={self.pattern}"
    
    def __call__(self, content):
        # return re.findall(self.pattern, content, re.DOTALL)[0]
        return re.search(self.pattern, content, re.DOTALL).group(1).strip()

class WriteFile(Component):
    """Write codes from LLM into files.
    """
    def __init__(self, path: str):
        self.path = path
        self.to_print = f"path={self.path}"

    def __call__(self, content):
        with open(self.path, "w") as f:
            f.write(content)
        return f"Successfully written to file {self.path}."

class ReadFile(Component):
    """Read from files.
    """

class INCAROverride(Component):
    """Override INCAR file with provided arguments.
    """
    def __init__(self, config: dict):
        self.config = config
        self.to_print = f"config={self.config}"
    def __call__(self, content):
        buffer = ""
        current_config = {}
        for line in content.split("\n"):
            if line.startswith("#"):
                continue
            if line.strip() == "":
                continue
            key, value = line.split("=")
            key = key.strip()
            value = value.strip()
            current_config[key] = value
        for key, value in self.config.items():
            if key in current_config:
                buffer += f"# {key} = {current_config[key]} # original generated\n"
            current_config[key] = str(value) + " # overridden"
        for key, value in current_config.items():
            buffer += f"{key} = {value}\n"
        return buffer

# TODO: hand-crafted or vasp command
class INCARValidaton(Component):
    """Validate INCAR file.
    """
    def __call__(self, content):
        return "INCAR file is valid."

class Command(Component):
    """Execute hand-crafted or context command.
    """
    def __init__(self, cwd: str, command: str = None):
        self.cwd = cwd
        self.command = command
        self.to_print = f"command={self.command}"

    def __call__(self, content):
        if self.command: command = self.command
        else: command = content
        result = subprocess.run(command, cwd=self.cwd, shell=True, capture_output=True, text=True, timeout=TIMEOUT)
        return {"returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr}

class GetReturncode(Component):
    """Get returncode from command execution.
    """
    def __call__(self, content):
        return content["returncode"]

class GetStdout(Component):
    """Get stdout from command execution.
    """
    def __call__(self, content):
        return content["stdout"]

class GetStderr(Component):
    """Get stderr from command execution.
    """
    def __call__(self, content):
        return content["stderr"]