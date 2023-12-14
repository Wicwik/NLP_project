def green(s): return f"\033[92m{s}\033[00m" 

def bold(s): return f"\033[1m{s}\033[0m"

def blue(s): return f"\033[94m{s}\033[00m" 

def start(f): print(bold(blue(f"Test {f.split('/')[-1].split('.')[0]} started...")))

def passed(f): print(bold(green(f"Test {f.split('/')[-1].split('.')[0]} passed.")))