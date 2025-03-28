# DDL-DAPPERFL Setup and Dataset Download

This repository includes two scripts to set up the development environment and download the OfficeCaltech dataset.

## 1. Environment Setup

### Prerequisites
- You have [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/) installed and available on your system.
- You have cloned this repository locally.

### Steps
1. **Navigate** to the project directory:
   ```bash
   cd DDL-DAPPERFL
   ```
2. **Run `setup.sh`**:
   ```bash
   ./setup.sh
   ```
   This will:
   1. Create a new conda environment named `ddl-env` with Python 3.9.
   2. Install `openmpi` and `compilers` from `conda-forge`.
   3. Use pip to install all the packages specified in `requirements.txt`.
   4. Initialize conda if needed and activate the `ddl-env`.

**Note**: If your shell does not have execute permission for `setup.sh`, run:
```bash
chmod +x setup.sh
```
then re-run:
```bash
./setup.sh
```

**Important**: After `setup.sh` finishes, **make sure** your new environment is active:
```bash
conda activate ddl-env
```

## 2. Downloading the OfficeCaltech Dataset

Once your environment is set up, you can download the dataset by running:
```bash
./download_office_caltech.sh
```
(This script is named `download_office_caltech.sh`, but you called it above as `download_dataset.sh` – ensure you match the correct filename.)

### What the Script Does
1. Creates a `resources` folder if it doesn’t already exist.
2. Uses `gdown` to download the OfficeCaltechDomainAdaptation archive from Google Drive by ID.
3. Extracts the tarball into `./resources`.
4. Removes any `._` files (macOS resource forks).
5. Copies the extracted `images/` into a new folder: `./resources/Office_Caltech_10`.
6. Cleans up by removing the original tarball and extracted folder.

At the end, you’ll have **all images** in:
```
./resources/Office_Caltech_10
```

## 3. RUN

- **you need to have you wandb token**:
  ```bash
  python ./src/main.py --model dapperfl --dataset fl_officecaltech  --backbone resnet18
    ```
