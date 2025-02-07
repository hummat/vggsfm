# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


from setuptools import setup, find_packages

# Read the long description from the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r") as f:
    dependencies = f.read().split("\n")

setup(
    name="vggsfm",
    version="2.0.0",
    author="Jianyuan Wang",
    description="A package for the VGGSfM project.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/facebookresearch/vggsfm.git",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=dependencies,
    package_data={"vggsfm": ["cfgs/*.yaml"]},
    entry_points={"console_scripts": ["vggsfm-image=vggsfm.demo:demo_fn",
                                      "vggsfm-video=vggsfm.video_demo:demo_fn"]}
)
