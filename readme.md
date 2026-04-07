# Statistical Inference Learning Tool in Python

## About the Project
This application was developed as the final project for the Data Science Master’s Programme at the Polytechnic of Leiria (Politécnico de Leiria), Portugal. 

The application is inspired by GUI-driven statistical software available in the R ecosystem, such as R Commander (Rcmdr). 
The primary objective of this project is to bridge the gap between graphical statistical analysis and Python programming by providing an interactive Graphical User Interface (GUI) dedicated to learning inferential statistics.

Designed as an educational tool, the platform allows users to upload datasets and select various statistical tests through an intuitive interface. Upon execution, the application not only displays the statistical results but also provides the exact Python code used behind the scenes. This dual-output approach empowers students and researchers to perform robust statistical analyses while simultaneously learning how to implement them using Python's scientific libraries.

## Key Features
* **Interactive GUI:** Easy-to-use interface built with Streamlit.
* **Dataset Handling:** Seamless upload and parsing of custom datasets.
* **Statistical Testing:** A comprehensive suite of inferential statistics tests (e.g., proportion tests, independence tests, confidence intervals).
* **Code Generation:** Real-time display of the reproducible Python code corresponding to every executed test.

## Getting Started

### Prerequisites
Make sure you have [Python 3.8+](https://www.python.org/downloads/) installed on your machine.

### Installation

1. **Clone the repository** (if applicable):
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Create a virtual environment** (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install the required dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application
To start the application, run the following command in your terminal:
```bash
streamlit run app.py
```