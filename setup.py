from setuptools import setup

setup(
    name="reap_gsf",
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    url="https://github.com/ausseabed/reap-gsf",
    author="AusSeabed",
    description="Prototype module for unpacking the contents of a GSF file",
    keywords=[
        "bathymetry",
        "GSF",
    ],
    # packages=find_packages(),
    packages=["reap_gsf"],
    install_requires=[
        "numpy",
        "attrs",
        "pandas",
        "structlog",
    ],
    license="Apache",
    zip_safe=False,
)
