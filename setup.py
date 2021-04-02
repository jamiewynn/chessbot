import os

import setuptools

repository_dir = os.path.dirname(__file__)
with open(os.path.join(repository_dir, "requirements.txt")) as f:
    requirements = f.readlines()

setuptools.setup(
    name="chessbot",
    version='0.0.0',
    package_dir={"": "src"},
    packages=setuptools.find_packages("src"),
    setup_requires=['wheel'],
    install_requires=requirements,
)
