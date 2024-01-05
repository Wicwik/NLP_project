import subprocess
import os
import re

p = re.compile("^[0-9]{2}_test")

files = [s for s in sorted(os.listdir("tests")) if p.match(s)]

for f in files:
    subprocess.run(["python", f"tests/{f}"])
