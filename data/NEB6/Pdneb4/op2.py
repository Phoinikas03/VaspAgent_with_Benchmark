# -*- coding: utf-8 -*-
import subprocess
import sys
import os

env_copy = os.environ.copy()
env_copy.pop("LD_PRELOAD", None)

directory = sys.argv[1]

# 进入目录
os.chdir(directory)

# 运行命令：复制 POTCARagpd 到 POTCAR 并执行 run.sh
result = subprocess.run(
    "bash -c 'cp POTCARpdch POTCAR && bash ./run.sh'",
    shell=True,
    capture_output=True,
    text=True,
    timeout=999999,
    env=env_copy
)


# 输出命令执行的标准输出和错误输出
sys.stdout.write(result.stdout)
sys.stderr.write(result.stderr)

# 退出程序，返回命令的返回码
sys.exit(result.returncode)