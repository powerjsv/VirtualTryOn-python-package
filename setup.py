from setuptools import setup, find_packages

setup(
    name="luxuryvtontest",
    version="0.0.11",
    description="This is a test for my python package upload to pypi",
    author="powerjsv",
    author_email="powerjsv12@gmail.com",
    url="https://github.com/powerjsv/jsv_package_test",
    install_requires=[
        "torch==2.0.1+cu118",
        "torchaudio==2.0.2+cu118",
        "torchvision==0.15.2+cu118",
        "torchtriton==2.0.0",
        "pytorch-cuda==11.8",
        "accelerate==0.25.0",
        "torchmetrics==1.2.1",
        "transformers==4.36.2",
        "diffusers==0.25.0",
        "einops==0.7.0",
        "bitsandbytes==0.39.0",
        "scipy==1.11.1",
        "opencv-python",
        "gradio==4.24.0",
        "fvcore",
        "cloudpickle",
        "omegaconf",
        "pycocotools",
        "basicsr",
        "av",
        "onnxruntime==1.16.2",
    ],
    packages=find_packages(exclude=[]),
    keywords=["vton", "powerjsv", "toy project", "pypi"],
    python_requires=">=3.10",
    package_data={},
    zip_safe=False,
    classifiers=[
        "Programming Language :: Python :: 3.10"
        "Programming Language :: Python :: 3.11",
    ],
)
