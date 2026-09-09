# install.py
import subprocess
import sys

### SET THIS to false to rely on CPU builds
useGPU = True

gpuSuffix = "+cu121" if useGPU else ""

def install(packagelist, flags=None):
    flags = flags or []
    subprocess.check_call([sys.executable, "-m", "pip", "install", *flags, *packagelist])


install(["numpy==1.26.4","transformers==4.57.6"])     # should already be satisfied with package install

install(["torch==2.3.1"+gpuSuffix, "torchaudio==2.3.1"+gpuSuffix, "torchvision==0.18.1"+gpuSuffix],
        flags="--index-url https://download.pytorch.org/whl/cu121".split())

install([
        "retico-core @ git+https://github.com/retico-team/retico-core@39e9957ee008b3e13ab4248040929f29056fb4ad",
        "retico-huggingfacelm @ git+https://github.com/youngmb/retico-huggingfacelm",
        "retico-speechbraintts @ git+https://github.com/youngmb/retico-speechbraintts@course-fix-cpu",
        "retico-whisperasr @ git+https://github.com/retico-team/retico-whisperasr@cf568c2a8b746558ddd71b2bf6cb58ba1141d175",
])