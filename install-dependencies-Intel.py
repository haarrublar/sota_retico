# install.py
import subprocess
import sys


def install(packagelist, flags=None):
    flags = flags or []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *flags, *packagelist])

## compatible versions with MacBook Pro 2.4 GHz 8-Core Intel Core i9
install(["numpy==1.26.4","transformers==4.44.2"])
install(["torch==2.2.2", "torchaudio==2.2.2", "torchvision==0.17.2"],
        flags="--index-url https://download.pytorch.org/whl/".split())
install([
        "retico-core @ git+https://github.com/retico-team/retico-core@39e9957ee008b3e13ab4248040929f29056fb4ad",
        "retico-huggingfacelm @ git+https://github.com/youngmb/retico-huggingfacelm",
        "retico-speechbraintts @ git+https://github.com/youngmb/retico-speechbraintts@course-fix-cpu",
        "retico-whisperasr @ git+https://github.com/retico-team/retico-whisperasr@cf568c2a8b746558ddd71b2bf6cb58ba1141d175",
])
