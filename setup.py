from setuptools import setup, find_packages

setup(
    name="luxuryvtontest",
    version="0.0.11",
    description="This is a test for my python package upload to pypi",
    author="powerjsv",
    author_email="powerjsv12@gmail.com",
    url="https://github.com/powerjsv/jsv_package_test",
    install_requires=[
        "tqdm",
        "pandas",
        "scikit-learn",
    ],
    packages=find_packages(exclude=[]),
    keywords=["vton", "powerjsv", "toy project", "pypi"],
    python_requires=">=3.6",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)
