from setuptools import setup
from setuptools_rust import Binding, RustExtension

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='oddity',
    version='0.1.0',
    author='Lleyton Ariton',
    author_email='lleyton.ariton@egmail.com',
    description='Time Series Anomaly Detection',
    long_description=long_description,
    url='https://github.com/Lleyton-Ariton/oddity',
    install_requires=['numpy'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Rust",
        "License :: OSI Approved :: GPLv3 License",
        "Operating System :: OS Independent",
    ],
    rust_extensions=[RustExtension('odditylib.oddity', binding=Binding.PyO3)],
    packages=['odditylib'],
    include_package_data=True,
    zip_safe=False,
)
