# Gender Identification — Group Assignment

## Purpose
- This repository contains notebooks and a Colab demo for a student assignment focused on gender identification from chest X-ray images. The main training notebook is [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb). A Colab demo script is provided in [group-assignment-demo-colab.ipynb](group-assignment-demo-colab.ipynb) to run inference with a trained model and the Gemini API.

## Contents
- [group-assignment-identify-gender.ipynb](group-assignment-identify-gender.ipynb) — Primary training notebook using cleaned datasets in `./datasets/`.
- [group-assignment-identify-gender-no-data-cleaning.ipynb](group-assignment-identify-gender-no-data-cleaning.ipynb) — Training without data cleaning (comparison experiment).
- [group-assignment-identify-pathologies.ipynb](group-assignment-identify-pathologies.ipynb) — Experimental notebook attempting pathology identification.
- [group-assignment-explainable-ai-xai.ipynb](group-assignment-explainable-ai-xai.ipynb) — Evaluation and explainability analysis for the gender model.
- [group-assignment-demo-colab.ipynb](group-assignment-demo-colab.ipynb) — Demo app to run inference using a trained model and the Gemini API.
- `datasets/` — Preprocessed CSV files used by the notebooks.
- `models/` — Saved model files produced by training (example: `.keras`).

## Dataset
- Raw source: NIH Chest X-rays (Kaggle): https://www.kaggle.com/datasets/nih-chest-xrays/data
- This repo uses cleaned CSVs placed in `datasets/` (see files in that folder). The notebooks expect those CSVs and the referenced image files to be present when training.

# Guidance for Running the Notebooks and Demo (Gender Identification)
## Kaggle Notebook Setup (training)
1. Open a new Kaggle Notebook and name it appropriately (e.g., "Gender Identification Training").
2. Click `+ Add Input` and search for "NIH Chest X-rays", click "+" to attach the dataset to your notebook.
3. Click `Upload` and upload the cleaned CSV files from the `datasets/` folder in this repo, name the dataset title appropriately.
4. Click `File` → `Import Notebook` to import the notebooks from Github or upload them directly. 
5. Update file paths in the notebook to point to the correct locations of the CSVs (mainly for the variables `train_df`, `valid_df`, `test_df` and `csv_search`).
6. Set the runtime to use a GPU: `Session options` → `Accelerator` → `GPU T4 x2`.
7. Click `Run All` in the notebook to execute the training process or Click `Save Version` → `Save & Run All (Commit)` to save a version and run at backend without notebook interaction.
8. Monitor the output cells for training progress, or check the `Logs` tab for background execution logs if running a version.
9. After training completes, the model files will be saved in the `models/` folder or in the output in the notebook. You can download the model for use in the Colab demo.

Notes and tips:
- The model training will take about 3 - 4 hours on Kaggle's GPU. You can monitor progress in the output cells or logs.

## Colab Demo Setup (inference)
1. Open Google Colab and click `File` → `Open Notebook`, then choose `Upload` to upload the demo notebook.
2. Set up the runtime environment:
   - Click `Runtime` → `Change runtime type`.
   - Under `Hardware accelerator`, select `T4 GPU` and click `Save`.
3. Add Secrets:
   - Click `Secrets` in the left sidebar.
   - Add a new secret called `GEMINI_API_KEY` and paste your Gemini API key as the value. You can obtain a Gemini API key from the Google AI Studio. We can use a free tier key for this demo, but ensure you have it set up before running the demo.
   - Enable the Notbook Access.
4. Upload the trained model file (e.g., `model-uniq-frist-image-13-95-yrs.keras`) to the Colab environment:
   - Click the folder icon on the left sidebar to open the file explorer. You may need to wait for seconds for the file explorer to load.
   - Click the upload icon and select your trained model file from your local machine.
   - After uploading, note the file path (e.g., `/content/model-uniq-frist-image-13-95-yrs.keras`) and update the demo script to point to this model file if necessary.
5. Click `Run Cell` button or `Run All` to execute the demo script. The demo will load the model, run inference on sample images, and display results.

# Explore other notebooks if you are interested in:
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
- Demo script: [group-assignment-demo-colab.ipynb](group-assignment-demo-colab.ipynb)

