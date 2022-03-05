from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

# package requirements
with open("requirements.txt", encoding='utf-8') as f:
    requirements = [l.strip() for l in f.readlines() if l]

# package version
__version__ = None
with open('rsnaped/__init__.py', encoding='utf-8') as f:
    for row in f:
        if row.startswith('__version__'):
            __version__ = row.strip().split()[-1][1:-1]
            break

setup(
    name = "FISH_processing",
    version = __version__,
    author = "Luis Aguilera, Joshua Cook, Brian Munsky",
    author_email = "luisubald@gmail.com",
    description = ("Python codes to analyze FISH images using BIG-FISH and Cellpose."),
    long_description=long_description,
    long_description_content_type="text/markdown",
    license = "MIT",
    keywords = "FISH image processing",
    url = "https://github.com/MunskyGroup/FISH_Processing",
    package_dir ={'src':'src'},
    packages=find_packages(exclude=['docs','dataBases','notebooks','tests','__pycache__','.gitignore']),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6"
    ],
    python_requires='>=3.6',
)
