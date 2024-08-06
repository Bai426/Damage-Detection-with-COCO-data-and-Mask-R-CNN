# Damage-Detection-with-COCO-data-and-Mask-R-CNN
Install mmdetection

a. Create a conda virtual environment and activate it.
conda create -n open-mmlab python=3.7 -y
conda activate open-mmlab

b. Install PyTorch and torchvision following the official instructions, e.g.,
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

c. Clone the mmdetection repository.
git clone https://github.com/open-mmlab/mmdetection.git
cd mmdetection

d. Install build requirements and then install mmdetection. (We install pycocotools via the github repo instead of pypi because the pypi version is old and not compatible with the latest numpy.)
pip install -r requirements/build.txt
pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI"
pip install -v -e .  # or "python setup.py develop"
At last I also installed the optional files (To use optional dependencies like albumentations and imagecorruptions either install them manually):
pip install -r requirements/optional.txt

