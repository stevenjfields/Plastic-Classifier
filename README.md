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