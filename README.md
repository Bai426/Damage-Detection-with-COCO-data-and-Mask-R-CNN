# Damage (spalling and cracks)-Detection-with-COCO-data-and-Mask-R-CNN
1. Install mmdetection

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

2. Data preparation
   
a. coco-annotator (https://github.com/jsbroks/coco-annotator) was used to label the data. It was run with a docker. So please follow the instruction of coco-annotator to install and implement the tools for data labeling.

b. Most images were collected from the searching from website, where image sizes vary from 300x300 to 4600x4600. But some are from Dr. Chul Min Yeum's data (about 1000 images).

c. The data format is COCO.

3. Training

a. Configuration files for training spalling and cracking detection are in this path: /mmdetection/configs/cracking_spalling/ .

b. Datasets (StructureCrackDataset) are available here: https://github.com/Bai426/StructureCrackDataset.

c. Modified ./tools/train.py with the paths of configuration file, images, and annotation for training. The hyperparameters should be fine tuned here.

4. Testing

Testing file is in ./tools/test/py.

Please read the following papers for more information. If you think this repo is useful to your research, please cite them:

@Article{bai-2021-isprs,
  title={Detecting Cracks and Spalling Automatically in Extreme Events by End-to-end Deep Learning Frameworks},
  author={Bai, Yongsheng and Sezen, Halil and Yilmaz, Alper},
  journal={ISPRS Annals of the Photogrammetry, Remote Sensing and Spatial Information Sciences},
  volume={2},
  pages={161--168},
  year={2021},
  publisher={Copernicus GmbH}
}

@article{bai2022engineering,
  title={Engineering deep learning methods on automatic detection of damage in infrastructure due to extreme events},
  author={Bai, Yongsheng and Zha, Bing and Sezen, Halil and Yilmaz, Alper},
  journal={Structural Health Monitoring},
  pages={14759217221083649},
  year={2022},
  publisher={SAGE Publications Sage UK: London, England}
}
