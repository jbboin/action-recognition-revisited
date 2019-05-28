# action-recognition-revisited

**Recurrent Neural Networks for Person Re-identification Revisited**

By Jean-Baptiste Boin, in collaboration with Andr&eacute; Araujo and Bernd Girod

Image, Video and Multimedia Systems Group, Stanford University


## Setup

**Prerequisite**:
- Caffe (with Python bindings). Make sure that the installation path is included in the `$PYTHONPATH`.

**Step 1**: Clone the repository.

    $ git clone https://github.com/jbboin/action-recognition-revisited.git

**Step 2**: Copy the `config.example.py` file to your own `config.py` file that will contain your system-dependent paths that the project will use.

    $ cp config.example.py config.py

You will need to populate this new file with your own desired paths. (**NOTE:** The paths should be absolute paths.)
- `DATASET_DIR`: Directory that will contain the dataset (RGB and optical flow images), as well as the CNN features extracted from these images.
- `LRCN_MODELS_DIR`: Directory where the weights for the models trained by Donahue et al. will be stored.

**Step 3**: Download the [weights for the trained LRCN and single-frame (baseline) models](https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video_weights.html) to your local directory specified in `LRCN_MODELS_DIR`. You will need to download four files (Single frame / LSTM ; RGB / flow). Do not modify the file names.

**Step 4**: Follow the instructions given [here](https://people.eecs.berkeley.edu/~lisa_anne/LRCN_video) to generate the RGB and flow images that will be used as inputs to the models. We recommend directly downloading the extracted images that the authors provide. The link is found on the page we linked. At the end of this step, your directory located at `DATASET_DIR` should contain two directories named `frames` and `flow_images`.


## Extract frame features

The LCRN networks are composed of two distinct components: the CNN part that extract features from each frame independently, and the LSTM part that pools them together. In order to prevent redundant work by extracting the frame features several times, here we first extract and save the frame features with the CNNs. These features will then be used to evaluate our different architectures and test conditions.

You should run the following four scripts that each extract the frame features for a different network:

    $ ./run_extract_baseline_RGB.sh 
    $ ./run_extract_baseline_flow.sh
    $ ./run_extract_lstm_RGB.sh
    $ ./run_extract_lstm_flow.sh

Each script will generate a sub-directory in `DATASET_DIR` containing `.h5` files that store the extracted features for the test part of the dataset.


## Run evaluation

Now we can evaluate the baseline, LSTM and FSTM networks under the different conditions. Each condition has a corresponding pair of script (for RGB and optical flow images). We give the correspondence between the conditions and the scripts here:

- LSTM:
  - `run_evaluate_lstm_RGB.py`
  - `run_evaluate_lstm_flow.py`
- Baseline:
  - `run_evaluate_baseline_RGB.py`
  - `run_evaluate_baseline_flow.py`
- LSTM-shuffled:
  - `run_evaluate_lstm-shuffled_RGB.py`
  - `run_evaluate_lstm-shuffled_flow.py`
- LSTM-mean:
  - `run_evaluate_lstm-mean_RGB.py`
  - `run_evaluate_lstm-mean_flow.py`
- FSTM:
  - `run_evaluate_fstm_RGB.py`
  - `run_evaluate_fstm_flow.py`
- FSTM-mean:
  - `run_evaluate_fstm-mean_RGB.py`
  - `run_evaluate_fstm-mean_flow.py`

Each script can be run with Python directly (e.g. by running the command `python run_evaluate_lstm_RGB.py`) and will evaluate the specified condition on the test split of the dataset. The obtained accuracy will be printed at the end of the execution.
