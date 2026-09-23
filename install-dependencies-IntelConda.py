import subprocess, sys

def conda_install(packages, channels=None):
    cmd = ["conda", "install", "-y", "-p", sys.prefix]
    for c in (channels or []):
        cmd += ["-c", c]
    subprocess.check_call(cmd + packages)

def install(packagelist, flags=None):
    flags = flags or []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *flags, *packagelist])

# Same as the website: conda install pytorch==2.2.2 torchvision==0.17.2 torchaudio==2.2.2 -c pytorch
# numpy is added so conda keeps it at 1.x when solving
conda_install(["pytorch==2.2.2", "torchvision==0.17.2", "torchaudio==2.2.2",
               "numpy=1.26.4"], channels=["pytorch"])

# Pure-Python and GitHub packages go through pip, after conda
install(["transformers==4.44.2"])
install([
    "retico-core @ git+https://github.com/retico-team/retico-core@39e9957ee008b3e13ab4248040929f29056fb4ad",
    "retico-huggingfacelm @ git+https://github.com/youngmb/retico-huggingfacelm",
    "retico-speechbraintts @ git+https://github.com/youngmb/retico-speechbraintts@course-fix-cpu",
    "retico-whisperasr @ git+https://github.com/retico-team/retico-whisperasr@cf568c2a8b746558ddd71b2bf6cb58ba1141d175",
])

import importlib.metadata as md
for pkg in ["numpy", "torch", "torchaudio", "torchvision", "transformers"]:
    print(pkg, md.version(pkg))