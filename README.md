# From Cracked Accounts to Fake IDs 

## Development setup

The development setup consists of two parts.
The first part is needed for running the project.
The second part is needed for committing changes to the project.

### 1. Create the environment

This project uses `conda` and `poetry` for dependency management and building.
The setup is done as follows:

1. Create a new conda environment

    ```bash
    conda env create -f env.yml
    ```

2. Activate the environment

    ```bash
    conda activate data-2022
    ```

3. Tell poetry to not create a virtualenv

    ```bash
    poetry config --local virtualenvs.create false
    ```

4. Install all dependencies

    ```bash
    poetry install
    ```

5. Extract the data 

    ```bash
    tar -xvf data.tar.xz
    ```
6. Download the pre-trained fasttext embeddings:
   https://drive.google.com/drive/folders/1a9llDhoM6zD-sOKiM0AdSxDYq2-15PJD and put them in data/raw
    
7. Run the experiments
   
   ```bash
   dvc repro
   ```