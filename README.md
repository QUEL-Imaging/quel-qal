![Version](https://img.shields.io/badge/version-0.2.4-blue)
![Python Version](https://img.shields.io/badge/python-3.12-green)
[![PyPI](https://img.shields.io/pypi/v/quel-qal?color=green)](https://pypi.org/project/quel-qal/)
[![license](https://img.shields.io/badge/license-AGPL%20V3-blue)](https://github.com/QUEL-Imaging/quel-qal/blob/main/LICENSE)
# QUEL-QAL: QUEL Quantitative Analysis Library
Welcome to **quel-qal**! This repository contains Python code for analyzing images of QUEL Imaging's fluorescence phantoms and obtaining relevant metrics about the capabilities of your fluorescence imaging system. Find more information about our fluorescence targets [on our website](https://shop.quelimaging.com/resources/). Documentation on how to use **quel-qal** to analyze images can be found in the `doc` folder and in the [Wiki](https://github.com/QUEL-Imaging/quel-qal/wiki).

## Getting Started
This guide will help you get started with setting up and using **quel-qal**. Follow the steps below to install directly from PyPI, or clone this repository locally. Conda installation is not currently supported.

### Prerequisites
- **Quel-qal** requires Python version ≥ 3.12.
- It is also recommended (though not required) to set up a separate virtual environment to install to. Create and activate a new virtual environment as follows.</br></br>

    On macOS/Linux:
    ```
    python3 -m venv env
    source env/bin/activate
    ```
    On Windows:
    ```
    python -m venv env
    env\Scripts\activate
    ```

### PyPI Installation
You can install **quel-qal** directly from PyPI using:
```
pip install quel-qal
```
This will install **quel-qal** along with all its dependencies.

### GitHub Installation
Alternatively, you can install the repository from GitHub by following these steps:
1. **Clone the repository:**
    ```
    git clone https://github.com/QUEL-Imaging/quel-qal.git
    cd quel-qal
    ```
2. **Set up a virtual environment (optional):**</br>
    Create and activate a virtual environment as described above.
3. **Install dependencies:**</br>
    You can install the required dependencies using `pip`:
    ```
    pip install -r requirements.txt
    ```
4. **Install quel-qal:**</br>
    ```
    pip install -e .
    ```

## License

### Source Code
The source code in this repository is licensed under the [GNU Affero General Public License v3.0 (AGPL-3.0)](https://www.gnu.org/licenses/agpl-3.0.html). 

### Documentation
The contents of the `doc` folder in this repository are licensed under the [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). 

### Notes
- If you contribute to this repository, you agree that your contributions to the source code will be licensed under AGPL-3.0 and your contributions to the documentation will be licensed under CC BY 4.0.
- Please ensure compliance with both licenses when using or modifying the content of this repository.

## Funding
This work is partially funded by the NIH and ARPA-H:
- **NIBIB** Grants R43/44 EB029804
- **NCI** Contract 75N91021C00035
- **NCI/ARPA-H** Contract 75N91023C00052