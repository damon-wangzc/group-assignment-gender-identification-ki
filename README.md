# Gender Identification — Group Assignment

## Purpose
- This repository contains notebooks and a Colab demo for a student assignment focused on gender identification from chest X-ray images. The main training notebook is [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb). A Colab demo script is provided in [group-assignment-demo-colab.py](group-assignment-demo-colab.py) to run inference with a trained model and the Gemini API.

## Contents
- [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb) — Primary training notebook using cleaned datasets in `./datasets/`.
- [group-assignment-identify-gender-no-data-cleaning.ipynb](group-assignment-identify-gender-no-data-cleaning.ipynb) — Training without data cleaning (comparison experiment).
- [group-assignment-identify-pathologies.ipynb](group-assignment-identify-pathologies.ipynb) — Experimental notebook attempting pathology identification.
- [group-assignment-explainable-ai-xai.ipynb](group-assignment-explainable-ai-xai.ipynb) — Evaluation and explainability analysis for the gender model.
- [group-assignment-demo-colab.py](group-assignment-demo-colab.py) — Demo app to run inference using a trained model and the Gemini API.
- `datasets/` — Preprocessed CSV files used by the notebooks.
- `models/` — Saved model files produced by training (example: `.keras`).

## Dataset
- Raw source: NIH Chest X-rays (Kaggle): https://www.kaggle.com/datasets/nih-chest-xrays/data
- This repo uses cleaned CSVs placed in `datasets/` (see files in that folder). The notebooks expect those CSVs and the referenced image files to be present when training.

## Kaggle Notebook Setup (training)
1. Open a new Kaggle Notebook and attach the NIH Chest X-rays dataset (or upload the cleaned CSVs and image files).
2. Set the runtime to use a GPU: `Notebook Settings` → `Accelerator` → `GPU`.
3. Install any extra packages required via a top cell, for example:

```bash
!pip install -q tensorflow pandas numpy scikit-learn pillow matplotlib
```

4. Upload or copy the repository files into the notebook environment (or use a Kaggle dataset version of this repo).
5. Open [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb) and run cells from top to bottom. Ensure the `datasets/` path points to the location of the CSVs and images in the Kaggle environment.

Notes:
- If using the original NIH dataset on Kaggle, adapt file paths in the notebook to match Kaggle's dataset mount (usually `/kaggle/input/<dataset-name>/`).

## Colab Demo Setup (inference)
1. Open Google Colab and either upload the repository files or clone the repo:

```bash
!git clone <your-repo-url>
%cd ki
```

2. Set the runtime to GPU if using TensorFlow acceleration: `Runtime` → `Change runtime type` → `Hardware accelerator` → `GPU`.
3. Install runtime dependencies in a cell:

```bash
!pip install -q tensorflow pandas numpy pillow requests
```

4. Provide your trained model file in the `models/` folder. Example: `models/model-uniq-frist-image-13-95-yrs.keras`.
5. Gemini API: the demo script uses the Gemini API for enhanced inference/processing. Set your API key in Colab before running the demo, for example:

```python
import os
os.environ['GEMINI_API_KEY'] = 'YOUR_GEMINI_API_KEY'
```

6. Run the demo script:

```bash
!python3 group-assignment-demo-colab.py
```

Notes and tips:
- If the demo opens a local webserver, use Colab tunneling tools (e.g., `ngrok`) or adapt the script to run inline in Colab.
- If you prefer using a small interactive notebook instead of the `.py` demo, create a Colab notebook cell that imports the demo script and calls its inference functions.

## How to run training (high-level)
- Run [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb) in Kaggle (recommended) or locally/Colab after installing dependencies and ensuring `datasets/` and images are available.
- The notebook trains a Keras model and saves outputs to the `models/` folder. After training, use the saved `.keras` model with the demo script.

## Explainability & Evaluation
- Use [group-assignment-explainable-ai-xai.ipynb](group-assignment-explainable-ai-xai.ipynb) to analyze model performance, create visualizations (ROC, confusion matrix), and run explainability tools (Grad-CAM, saliency maps) included in the notebook.

## Notes and Caveats
- The NIH Chest X-rays dataset contains sensitive medical images; handle with care and follow any applicable data usage agreements.
- Model performance and fairness for gender identification from X-rays is an exploratory academic exercise — report limitations and ethical considerations in any writeup.

## Where to find files
- Notebooks: [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb), [group-assignment-identify-gender-no-data-cleaning.ipynb](group-assignment-identify-gender-no-data-cleaning.ipynb), [group-assignment-identify-pathologies.ipynb](group-assignment-identify-pathologies.ipynb), [group-assignment-explainable-ai-xai.ipynb](group-assignment-explainable-ai-xai.ipynb)
- Demo script: [group-assignment-demo-colab.py](group-assignment-demo-colab.py)

## Questions / Next steps
- Want me to add a `requirements.txt` or a short Colab notebook wrapper to run the demo interactively? Tell me which and I'll add it.
