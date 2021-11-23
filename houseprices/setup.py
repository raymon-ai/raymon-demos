from setuptools import setup, find_packages

setup(
    name="houseprices",
    version="0.0.1",
    author="Karel Vanhoorebeeck",
    description="Raymon API code",
    packages=find_packages(),
    install_requires=[
        "raymon==0.0.39",
    ],
)
