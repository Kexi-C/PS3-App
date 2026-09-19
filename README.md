# Train Condition Monitoring System

This is a Streamlit application for monitoring train condition. It integrates four subsystems—Door, ACV, Rail Corrugation, and SHM—into one interface. Users select a subsystem, upload sensor files, run the original model or analysis logic, and download a standard CSV result.

## Features

| Subsystem | Input | Output | Visualization |
| --- | --- | --- | --- |
| Door | One or more .csv recordings | door_predictions.csv with start_time, end_time, and prediction | 2×2 signal comparison for motor current, voltage, Back-EMF, and door speed against door-leaf position; blue means Normal and red means Abnormal resistance |
| ACV | One or more .xlsx train-case workbooks | acv_predictions.csv ranking cars by suspected refrigerant-leak likelihood | Car-ranking bars and cabin-temperature-over-time lines |
| Rail Corrugation | One or more .csv vibration recordings | rail_predictions.csv | Class counts, two-side vibration scatter plot, and per-car/per-rail-side vibration heatmap |
| SHM | One or more .csv, .txt, .dat, .npy, or .npz dynamic-stress files | shm_predictions.csv with one fatigue-damage value per file | Damage-by-file bars, stress waveform, and damage contribution by amplitude range |

Uploaded files are used only for the current analysis session and are not written back to the source files. Door and SHM production inference do not require users to upload answer files; the model and analysis entry points are supplied by the corresponding subsystem directories.

## Recommended project layout

For deployment, copy the current version to streamlit_app.py in the project root and keep the subsystem code and model files inside the same build context:

~~~text
.
├── streamlit_app.py
├── requirements.txt
├── Dockerfile
└── subsystems/
    ├── door/
    │   └── code/
    ├── acv/
    │   └── code/
    ├── rail_corrugation/
    │   ├── code/
    │   └── model/
    │       └── rail_model.joblib
    └── shm/
        └── code/
~~~

The application also supports some older layouts: a subsystem directory may be placed beside the entrypoint, or the subsystem code may be placed directly in the subsystem root. The recommended layout above is safer for deployment because it reduces the chance of omitting a model or original analysis module from the Docker build context.

If the current source file is named streamlit_app_latest(1).py, run this from the project root:

~~~bash
cp 'streamlit_app_latest(1).py' streamlit_app.py
~~~

If you do not copy the file, update the local start command and the Dockerfile so that both point to the actual entrypoint.

## Input formats

### Door

- Accepts one or more .csv files, such as Train.csv, Test.csv, or other continuous recordings.
- The first column, or a column named timestamp or datetime, should contain timestamps such as 2023-7-5-0-0-3-760.
- At least 17 columns are required. Extra columns are retained and do not automatically cause rejection.
- The recording should contain motor current, voltage, Back-EMF, door timing, control commands, switch states, and door-leaf-position signals.
- Train_Segments_Answer.csv is not required; production inference uses the saved Door model or adapter.

### ACV

- Use one .xlsx workbook per train case. Multiple cases can be uploaded together.
- The first worksheet is used, and the first row must contain the headers.
- The headers should include a time column and per-car parameter columns such as Car 03 - ACV Running Mode.
- openpyxl is required to read .xlsx files in the cloud deployment.

### Rail Corrugation

- Use one .csv file per recording.
- The recommended format contains approximately 10,000 rows and 129 numeric columns: column 1 is a 0/1 speed pulse, followed by vibration and shock signals for 8 cars and 8 axle-box positions.
- The application treats the sampling rate as 10 kHz. A small row-count deviation produces a warning but does not by itself reject the file.
- The output classes are Normal, Side I, and Side II.

### SHM

- Accepts .csv, .txt, .dat, .npy, and .npz files.
- The UI performs only an empty-file check. The original SHM module remains responsible for delimiters, metadata rows, and train/test filename conventions.
- The SHM subsystem should expose a batch or single-file inference interface such as predict_shm, predict_shm_batch, predict_files, or predict.
- If the original module uses a low-level function such as analyze(data, filename, model), the subsystem adapter must supply those arguments. The low-level function must not be mistaken for a single-argument prediction function.

## Run locally

~~~bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
~~~

If the latest filename is kept:

~~~bash
streamlit run 'streamlit_app_latest(1).py'
~~~

The direct dependencies should include at least:

~~~text
streamlit
pandas
numpy
altair
openpyxl
scipy
scikit-learn
joblib
~~~

openpyxl, model-loading dependencies, and all original subsystem dependencies must be declared in requirements.txt. The deployment must not rely on packages that happen to be installed in the local machine environment.

## Docker / Google Cloud Run

Example minimal Dockerfile:

~~~dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=8080
EXPOSE 8080

CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8080"]
~~~

Deploy from the project root:

~~~bash
gcloud run deploy <SERVICE_NAME> \
  --source . \
  --region us-central1 \
  --allow-unauthenticated
~~~

Local file changes do not automatically update Cloud Run. After changing requirements.txt, the Dockerfile, the entrypoint, or subsystem code, rebuild and redeploy the service; otherwise the online page may still be running an older image.

## Pre-deployment checks

~~~bash
python -m py_compile streamlit_app.py
python -c "import streamlit, pandas, numpy, altair, openpyxl"
~~~

Make sure the following files are not excluded by .gitignore or by the Docker build context:

- streamlit_app.py
- requirements.txt
- Dockerfile
- subsystems/door/
- subsystems/acv/
- subsystems/rail_corrugation/
- subsystems/shm/
- The Rail model file rail_model.joblib

## Troubleshooting

### ACV works locally but shows Cannot be opened as an Excel workbook after deployment

The cloud image usually does not contain openpyxl, or the service is still using an older image. Add openpyxl to requirements.txt and redeploy. Also verify that the Cloud Run Dockerfile and CMD point to the current entrypoint.

### The page shows SHM analysis failed for every uploaded file

First verify that the SHM subsystem files were copied into the image. Then inspect the function signature exposed by the adapter. In particular, do not call a low-level analyze() function that requires filename and model with only one argument. Use the production prediction entry point or perform the argument conversion in the adapter.

### The local page is correct but the online page is still old

Check whether the Dockerfile CMD still starts the old streamlit_app.py and whether the build context actually contains the latest entrypoint. Run the Cloud Run deployment again after making the change.

### The page says that an input file does not match the format

Read the Note column in the upload table. The UI performs lightweight format checks. For SHM, detailed parsing remains inside the original module so that valid files with metadata rows or non-standard delimiters are not rejected by a generic CSV pre-check.

## Git commit

Run the following commands from the real project root:

~~~bash
git status
git add README.md
git commit -m "docs: add application and deployment guide"
git push origin "$(git branch --show-current)"
~~~

If the project is not a Git repository yet, initialize Git and configure the real remote in the correct project root first. Do not treat the uploaded-copy directory as the production repository.
