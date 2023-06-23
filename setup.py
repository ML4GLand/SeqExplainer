from setuptools import setup, find_packages

setup(name = 'seqexplainer', packages = find_packages())

with open("README.md", "r") as readme_file:
    readme = readme_file.read()

requirements = []

setup(
    name="seqexplainer",
    version="0.1.0",
    author="Adam Klie",
    author_email="aklie@ucsd.edu",
    description="A tool for interpreting sequence input genomics PyTorch models",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/adamklie/SeqExplainer",
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
    ],
)
