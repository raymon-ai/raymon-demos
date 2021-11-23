from setuptools import setup, find_packages

"""
Research code for ovality detection.

"""
setup(
    name="retinopathy",
    version="0.0.1",
    author="Karel Vanhoorebeeck",
    description="Raymon API code",
    packages=find_packages(),
    install_requires=[
        "raymon==0.0.39",
    ],
)
