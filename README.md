# PyVirtry

## Description

PyVitry is an end-to-end Python package that provides simple and efficient application of Virtual Try On technology.
Virtual try-on is an innovative way for customers to visualize clothes, makeup, glasses, and accessories without putting them on physically. Although many research and commercial projects are in progress, there is no library that users can easily use end-to-end. Therefore, we developed a library that can execute virtual try on with a few lines of simple code without complex work. We utilized [IDM-VTON](https://github.com/yisol/IDM-VTON) which is very powerful and performs well. This model better expresses the texture and pattern of clothes.

## Installation and Usage

### Prerequisites

To install PyVirtry, you will need the following minimum specifications:
+ Python 3.10 or later
+ System RAM 17GB
+ GPU RAM 18.7GB
+ Disk 62GB

It is assumed that the Pytorch suitable for the user and the cuda version are compatible accordingly.

### Getting Started

__NOTE:__ It only currently provides synthetic results for the upper body.

1. Install PyVirtry library

```
pip install PyVirtry
```

2. Import the library and create an IDMVTON instance.

```
from PyVirtry.VirtualTryOn import models

model = models.IDMVTON()
```

3. You must specify 
+ human image path
+ garment image path
+ description of the garment
+ output image path

The description should include the texture, shape, form, type, etc. of the clothing. The more detailed it is, the more helpful it is in expressing the model.

```
result = model.start_tryon(human_img_path = 'Enter your HUMAN IMAGE PATH',
                           garm_img_path = 'Enter your GARMENT IMAGE PATH', 
                           garment_des = 'Enter the DESCRIPTION OF GARMENT',
                           output_path= 'Enter your OUTPUT PATH')

```

## Reference

This project is based on [IDM-VTON](https://github.com/yisol/IDM-VTON). We modified the code to make it work simply.  

## Licencse

We follow the [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode) license specified in IDM-VTON. Please note that this is not intended for commercial use.

## Getting Help

If you encounter any issues or have questions, please open an issue on our GitHub page.