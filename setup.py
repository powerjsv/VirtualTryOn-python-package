from setuptools import setup, find_packages

setup(
    name="moipg",
    version="0.0.8",
    description="This is a test for my python package upload to pypi",
    author="powerjsv",
    author_email="powerjsv12@gmail.com",
    url="https://github.com/powerjsv/jsv_package_test",
    install_requires=[
        "pip==23.3.1",
        "accelerate==0.25.0",
        "tqdm==4.66.1",
        "transformers==4.36.2",
        "diffusers==0.25.0",
        "einops==0.7.0",
        "bitsandbytes==0.39.0",
        "scipy==1.11.1",
        "opencv-python",
    ],
    packages=find_packages(exclude=[]),
    keywords=["vton", "powerjsv", "toy project", "pypi"],
    python_requires=">=3.10",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
)
