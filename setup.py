import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()
version = "0.10.0"
setuptools.setup(
    name="tkmatrix",
    version=version,
    author="M. Dévora-Pajares & F.J. Pozuelos",
    author_email="mdevorapajares@protonmail.com",
    description="ToolKit for Multi-phAse Transits Recovery from Injected eXoplanets",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/PlanetHunters/tkmatrix",
    packages=setuptools.find_packages(),
    include_package_data=True,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.10',
    install_requires=['argparse==1.4.0',
                        'beautifulsoup4==4.9.3',
                        'configparser==5.0.1',
                        "corner==2.1.0",
                        "ellc==1.8.5",
                        "lcbuilder==0.18.3",
                        "mock==4.0.3",
                        'pyparsing==2.4.7', # Matplotlib dependency
                        "seaborn==0.11.1",
                        'setuptools>=41.0.0',
                        "sklearn==0.0"
    ]
)
