# Instruction to run this repo
* clone the repository

```
python3 -m pip install --upgrade pip setuptools

pip install numpy cython

cd openpifpaf
python3 -m pip install --editable '.[train,test]'

time  python3 -m openpifpaf.train \
  --batch-size=8 \
  --basenet=mobilenetv1 \
  --head-quad=1 \
  --epochs=150 \
  --momentum=0.9 \
  --headnets pif paf\
  --lambdas 30 2 2 50 3 3\
  --loader-workers=16 \
  --lr=0.1 \
  --lr-decay 120 140 \
  --no-pretrain \
  --weight-decay=1e-5 \
  --update-batchnorm-runningstatistics \
  --ema=0.03
  --no-augmentation \

or simply run
sh train_mobilenet.sh
```


# Related Projects
* [openpifpaf](https://github.com/vita-epfl/openpifpaf): The base of this project
* [lightweighted openpose](https://github.com/Daniil-Osokin/lightweight-human-pose-estimation.pytorch) "Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose" which optimize OpenPose to achieve realtime demo on CPU
* 


# Citation

```
@InProceedings{kreiss2019pifpaf,
  author = {Kreiss, Sven and Bertoni, Lorenzo and Alahi, Alexandre},
  title = {PifPaf: Composite Fields for Human Pose Estimation},
  booktitle = {The IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  month = {June},
  year = {2019}
}
@inproceedings{osokin2018lightweight_openpose,
    author={Osokin, Daniil},
    title={Real-time 2D Multi-Person Pose Estimation on CPU: Lightweight OpenPose},
    booktitle = {arXiv preprint arXiv:1811.12004},
    year = {2018}
}
```


[CC-BY-2.0]: https://creativecommons.org/licenses/by/2.0/
