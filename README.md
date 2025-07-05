# phase-transformation-jmak
This repository implements isothermal phase-transformation kinetics modeling using the Johnson–Mehl–Avrami–Kolmogorov (JMAK) equation on steel TTT data.
## Contents
- `docs/theory.md`: Two-page theoretical background on the JMAK model and its application to ferrite, pearlite, and bainite transformations
- `docs/code_explanation.md`: Step-by-step explanation of each part of the code
- `src/jmak_model.py`: Main Python script
- `notebooks/analysis.ipynb`: Jupyter notebook combining code, results, and figures
- `data/TTT_data_Steel.csv` – raw input data
- `data/TTT_synthetic.xlsx` – generated synthetic curves & parameters

## Usage
1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Run `src/jmak_model.py` or open `notebooks/analysis.ipynb`

## License
MIT License
