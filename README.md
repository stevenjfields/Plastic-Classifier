# The_Dump
Plastic Classification using image processing through OpenCV's CDN module 

## Setup

First, make sure the latest version of Python is installed.

### Virtual Environment

Create a virtual enviornment in the project folder. The final folder in the path should be named Env.

```bash
python -m venv C:\Path\To\Virtual\Environment
```

### Install Dependencies

Activate the virtual environment and install dependencies:

```bash
C:\Path\To\Env\Scripts\activate
(Env) pip install -r requirements.txt
```

### Add necessary files

Add the necessary files into the resources folder. The .cfg and .names files are included, But the .weights file will need to be [downloaded](https://drive.google.com/open?id=1wjsJfALPRW9w1FvDqGvXg4aZbWUEO7ef).

## Running
To run this program first activate the virtual environment and then run the Plastic_Classifier_GUI.py file.

```bash
C:\Path\To\Env\Scripts\activate
(Env) python Plastic_Classifier_GUI.py
```

## Utilizing BoundingBoxes.txt

The output file gives a label and 4 values, Center X, Center Y, Width, and Height. To find the bounding boxes using these values, do as follows :

1. Subtract Width/2 from X and Height/2 from Y to get the bottom left corner of the box.
2. Add Width to X and Height to Y to get the top right corner of the box.

From plasticClassifier.py:

```python
#Lines 109 and 110. Values are later rounded in lines 52 and 53.
x = center_x - w / 2
y = center_y - h / 2

#Lines 54 and 55
x_plus_w = round(x+w)
y_plus_h = round(y+h)
```