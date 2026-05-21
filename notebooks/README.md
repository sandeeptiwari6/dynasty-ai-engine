# Notebooks
These are the notebooks used to for exploration, experimentation, and analysis. Most of these notebooks primarily serve to demonstrate how the code should be written.

In order to run the notebooks, run the following steps from the root directory:
1. If you haven't already, create the virtual environment, and activate it. If you've already done this, skip to step 2
    ```
    pipenv install --dev
    pipenv shell
    ```
2. The go to the `notebooks/` directory and then kick off jupyter
    ```
    jupyter lab --notebook-dir=$(pwd)/notebooks
    ```
