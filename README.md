# 2023-dtu-deep-learning-project

## Description
Group project in Fall course "Deep Learning" at DTU. Consists of a simple Streamlit front-end for Langchain RAG for a teaching-learning system. 

## Installing required packages
In the project folder, create a new virtual environment and activate it:
```
python -m venv .venv
```
Next, install project dependencies in the venv:
```
pip install -r "requirements.txt"
```
Finally, test that streamlit installation works:
```
streamtlit hello
```
Streamlit's `hello` app should appear in a new tab in a browser window. If installation of Streamlit fails, use the following guide: https://docs.streamlit.io/library/get-started/installation.

## Running the app
In the project folder, if the `venv` is not active, run the following command:
```
source .venv/bin/activate
```
Next, simply run:
```
streamlit run app.py
```
The app should now appear in a browser window. To deactive the app, press `ctrl-c` in the terminal. When finished with the environment, use the `deactivate` command.
