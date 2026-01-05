## Prerequisites

Make sure you have Python 3.10+ installed. You can check by running:
python --version

## Clone the Repository

git clone https://github.com/Kasperdejong/Python3d.git
cd Python3d

## Create a Virtual Environment

It is best practice to use a virtual environment so libraries don't mess up your system.

# On macOS / Linux:

python3 -m venv venv

Run this if you've already created it once.
source venv/bin/activate

# On Windows:

python -m venv venv
venv\Scripts\activate

## Install Dependencies

This project relies on MediaPipe, OpenCV, Flask, and Eventlet. Install them automatically using the requirements file:
pip install -r requirements.txt

## How to Run

Make sure your webcam is connected.
python treeperson.py

## Run and Autoreload on script change (do this if you are actively working on the code and want to test as you go)

run this in terminal. The filename has to be the one you're working on. You still need to reload the browser when applying a change ctrl + shift + r for hard reload

watchfiles "python handpuppets.py"
