# Quaternion Unity

<p align="center">
  <a href="https://github.com/34j/quaternion-unity/actions/workflows/ci.yml?query=branch%3Amain">
    <img src="https://img.shields.io/github/actions/workflow/status/34j/quaternion-unity/ci.yml?branch=main&label=CI&logo=github&style=flat-square" alt="CI Status" >
  </a>
  <a href="https://quaternion-unity.readthedocs.io">
    <img src="https://img.shields.io/readthedocs/quaternion-unity.svg?logo=read-the-docs&logoColor=fff&style=flat-square" alt="Documentation Status">
  </a>
  <a href="https://codecov.io/gh/34j/quaternion-unity">
    <img src="https://img.shields.io/codecov/c/github/34j/quaternion-unity.svg?logo=codecov&logoColor=fff&style=flat-square" alt="Test coverage percentage">
  </a>
</p>
<p align="center">
  <a href="https://python-poetry.org/">
    <img src="https://img.shields.io/badge/packaging-poetry-299bd7?style=flat-square&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAA4AAAASCAYAAABrXO8xAAAACXBIWXMAAAsTAAALEwEAmpwYAAAAAXNSR0IArs4c6QAAAARnQU1BAACxjwv8YQUAAAJJSURBVHgBfZLPa1NBEMe/s7tNXoxW1KJQKaUHkXhQvHgW6UHQQ09CBS/6V3hKc/AP8CqCrUcpmop3Cx48eDB4yEECjVQrlZb80CRN8t6OM/teagVxYZi38+Yz853dJbzoMV3MM8cJUcLMSUKIE8AzQ2PieZzFxEJOHMOgMQQ+dUgSAckNXhapU/NMhDSWLs1B24A8sO1xrN4NECkcAC9ASkiIJc6k5TRiUDPhnyMMdhKc+Zx19l6SgyeW76BEONY9exVQMzKExGKwwPsCzza7KGSSWRWEQhyEaDXp6ZHEr416ygbiKYOd7TEWvvcQIeusHYMJGhTwF9y7sGnSwaWyFAiyoxzqW0PM/RjghPxF2pWReAowTEXnDh0xgcLs8l2YQmOrj3N7ByiqEoH0cARs4u78WgAVkoEDIDoOi3AkcLOHU60RIg5wC4ZuTC7FaHKQm8Hq1fQuSOBvX/sodmNJSB5geaF5CPIkUeecdMxieoRO5jz9bheL6/tXjrwCyX/UYBUcjCaWHljx1xiX6z9xEjkYAzbGVnB8pvLmyXm9ep+W8CmsSHQQY77Zx1zboxAV0w7ybMhQmfqdmmw3nEp1I0Z+FGO6M8LZdoyZnuzzBdjISicKRnpxzI9fPb+0oYXsNdyi+d3h9bm9MWYHFtPeIZfLwzmFDKy1ai3p+PDls1Llz4yyFpferxjnyjJDSEy9CaCx5m2cJPerq6Xm34eTrZt3PqxYO1XOwDYZrFlH1fWnpU38Y9HRze3lj0vOujZcXKuuXm3jP+s3KbZVra7y2EAAAAAASUVORK5CYII=" alt="Poetry">
  </a>
  <a href="https://github.com/ambv/black">
    <img src="https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square" alt="black">
  </a>
  <a href="https://github.com/pre-commit/pre-commit">
    <img src="https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white&style=flat-square" alt="pre-commit">
  </a>
</p>
<p align="center">
  <a href="https://pypi.org/project/quaternion-unity/">
    <img src="https://img.shields.io/pypi/v/quaternion-unity.svg?logo=python&logoColor=fff&style=flat-square" alt="PyPI Version">
  </a>
  <img src="https://img.shields.io/pypi/pyversions/quaternion-unity.svg?style=flat-square&logo=python&amp;logoColor=fff" alt="Supported Python versions">
  <img src="https://img.shields.io/pypi/l/quaternion-unity.svg?style=flat-square" alt="License">
</p>

---

**Documentation**: <a href="https://quaternion-unity.readthedocs.io" target="_blank">https://quaternion-unity.readthedocs.io </a>

**Source Code**: <a href="https://github.com/34j/quaternion-unity" target="_blank">https://github.com/34j/quaternion-unity </a>

---

Unity-like and NumPy-compatible quaternion

## Installation

Install this via pip (or your favourite package manager):

`pip install quaternion-unity`

## Alternatives

- [clemense/quaternion\-conventions: An overview of different quaternion implementations and their chosen order: x\-y\-z\-w or w\-x\-y\-z?](https://github.com/clemense/quaternion-conventions)
- Specific to Quarternion/Rotation
  - [moble/quaternion: Add built-in support for quaternions to numpy](https://github.com/moble/quaternion): slerp, etc. is not implemented, operations are implemented as functions, not methods
  - [RoMa: A lightweight library to deal with 3D rotations in PyTorch. — RoMa latest documentation](https://naver.github.io/roma/#api-documentation)
  - [Quaternions — Spatial Maths package documentation](https://bdaiinstitute.github.io/spatialmath-python/func_quat.html#module-spatialmath.base.quaternions)
  - [the-guild-of-calamitous-intent/squaternion: Simple quaternion math library](https://github.com/the-guild-of-calamitous-intent/squaternion)
  - [konbraphat51/UnityQuaternionPy: UnityEngine.Quaternion from scratch in Python3](https://github.com/konbraphat51/UnityQuaternionPy): unity-like interface but not numpy-compatible
  - [translunar/pyquat: Fast unit quaternion code for Python written in C](https://github.com/translunar/pyquat): C implementation
  - [PhilJd/tf-quaternion: An implementation of quaternions for and written in tensorflow. Fully differentiable.](https://github.com/PhilJd/tf-quaternion): not actively maintained, differentiable
  - [ispamm/hTorch: Repository dedicated to Quaternion Neural Networks](https://github.com/ispamm/hTorch): not actively maintained, neural network
  - [satellogic/quaternions](https://github.com/satellogic/quaternions): not actively maintained
  - [quaternions/quaternions/quaternion.py at master · mjsobrep/quaternions](https://github.com/mjsobrep/quaternions/blob/master/quaternions/quaternion.py): not actively maintained
  - [KieranWynn/pyquaternion: A fully featured, pythonic library for representing and using quaternions](https://github.com/KieranWynn/pyquaternion): not actively maintained
  - [tinyquaternion/tinyquaternion/tinyQuaternion.py at master · rezaahmadzadeh/tinyquaternion](https://github.com/rezaahmadzadeh/tinyquaternion/blob/master/tinyquaternion/tinyQuaternion.py): not actively maintained
- General scientific computing
  - [Quaternion — Pyrr 0.10.3 documentation](https://pyrr.readthedocs.io/en/latest/api_quaternion.html)
  - [pytorch3d.transforms.rotation_conversions — PyTorch3D documentation](https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/transforms/rotation_conversions.html#standardize_quaternion): no quaternion class, only a few functions
  - [pytransform3d/pytransform3d/rotations/\_quaternion_operations.py at c45e817c4a7960108afe9f5259542c8376c0e89a · dfki-ric/pytransform3d](https://github.com/dfki-ric/pytransform3d/blob/c45e817c4a7960108afe9f5259542c8376c0e89a/pytransform3d/rotations/_quaternion_operations.py#L22): only a few functions
  - [Spatial Transformations (scipy.spatial.transform) — SciPy v1.14.1 Manual](https://docs.scipy.org/doc/scipy/reference/spatial.transform.html#): only a few functions
  - [quaternions — transforms3d 0.4 documentation](https://matthew-brett.github.io/transforms3d/reference/transforms3d.quaternions.html)
- Robotics
  - [Quaternions — autolab_core 1.1.0 documentation](https://berkeleyautomation.github.io/autolab_core/api/dual_quaternion.html#autolab_core.DualQuaternion.qr): only 5 methods
  - [PyBullet Quickstart Guide - Google Docs](https://docs.google.com/document/d/10sXEhzFRSnvFcl3XxNGhnD4N2SedqwdAvK3dsihxVUA/edit#heading=h.2ye70wns7io3): documentation is not clear
  - [dm_robotics/py/transformations at main · google-deepmind/dm_robotics](https://github.com/google-deepmind/dm_robotics/tree/main/py/transformations)

## Contributors ✨

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- prettier-ignore-start -->
<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- markdownlint-disable -->
<!-- markdownlint-enable -->
<!-- ALL-CONTRIBUTORS-LIST:END -->
<!-- prettier-ignore-end -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!

## Credits

This package was created with
[Copier](https://copier.readthedocs.io/) and the
[browniebroke/pypackage-template](https://github.com/browniebroke/pypackage-template)
project template.
