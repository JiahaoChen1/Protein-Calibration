This file contains the code for the paper "Calibrating the Reliability of AlphaFold2"

### Requirements
```console
Ubuntu 20.04.1
NVIDIA A40

conda create -n ProCal python=3.8
conda activate ProCal
pip install -r requirements.txt
```
Torch_scatter and torch_cluster libraries depend on the specific settings for the server. More details are in https://github.com/rusty1s/pytorch_scatter and https://github.com/rusty1s/pytorch_cluster. It takes about half an hour to install all the packages.

### Directory Structure
```console
--Protein-Calibration-main
  -CollectedData            # calibrate our collected data
    -PAE                    # calibrate PAE metric
     ...
    -pLDDT                  # calibrate pLDDT metric
     ...
  -ShortPeptideData         # calibrate peptide data
     ...
```

### Perform calibration on our collected data
For the data we collected, we implement calibration of the PAE and pLDDT metrics. For each experiment, we perform the random division of validation and test sets for five times. The calibration models were initialized randomly. Download our collected dataset into the folder: PAE/data or pLDDT/data Here we provide some examples in this folder.
#### Calibrate PAE
Firstly, go to the PAE folder:
```console
cd CollectedData/PAE
```
Calibrate PAE with RUCS-regression (random seed is 0 and split 80% data as the training set):
```console
python main.py --alg std_scaling --dataset aelddt --select ae --method reg --data_seed 0 --frac 0.8
```
Calibrate PAE with RUCS-NLL (random seed is 0 and split 80% data as the training set):
```console
python main.py --alg std_scaling --dataset aelddt --select ae --method nll --data_seed 0 --frac 0.8
```
Calibrate PAE with R<sup>2</sup>UCS-regression:
```console
python main.py --alg gemo_scaling --dataset aelddt --select ae --feature 2 --method reg --data_seed 0 --frac 0.8
```
Calibrate PAE with R<sup>2</sup>UCS-NLL:
```console
python main.py --alg gemo_scaling --dataset aelddt --select ae --feature 2 --method nll --data_seed 0 --frac 0.8
```
#### Calibrate pLDDT
Firstly, go to the pLDDT folder:
```console
cd CollectedData/pLDDT
```
Calibrate pLDDT with RUCS-regression:
```console
python main.py --alg std_scaling --dataset aelddt --select lddt --method reg --data_seed 0 --frac 0.8
```
Calibrate pLDDT with RUCS-NLL:
```console
python main.py --alg std_scaling --dataset aelddt --select lddt --method nll --data_seed 0 --frac 0.8
```
Calibrate pLDDT with R<sup>2</sup>UCS-regression:
```console
python main.py --alg gemo_scaling --dataset aelddt --select lddt --feature 2 --method reg --data_seed 0 --frac 0.8
```
Calibrate pLDDT with R<sup>2</sup>UCS-NLL:
```console
python main.py --alg gemo_scaling --dataset aelddt --select lddt --feature 2 --method nll --data_seed 0 --frac 0.8
```
### Perform calibration on the peptide data
For the peptide data, we implement calibration of the pLDDT metric. Obtain the peptide dataset and put it into the fold: data
```console
cd ShortPeptideData
```

Calibrate pLDDT with RUCS-regression:
```console
python main.py --alg std_scaling --dataset peptide --select lddt --method reg --data_seed 0 --frac 0.8
```
Calibrate pLDDT with RUCS-NLL:
```console
python main.py --alg std_scaling --dataset peptide --select lddt --method nll --data_seed 0 --frac 0.8
```
Calibrate pLDDT with R<sup>2</sup>UCS-regression:
```console
python main.py --alg gemo_scaling --dataset peptide --select lddt --feature 2 --method reg --data_seed 0 --frac 0.8
```
Calibrate pLDDT with R<sup>2</sup>UCS-NLL:
```console
python main.py --alg gemo_scaling --dataset peptide --select lddt --feature 2 --method nll --data_seed 0 --frac 0.8
```
Each command executes the training and testing of the corresponding calibration model, which takes less than 2 minutes to run on the toy data we provide and outputs the RMSE or ECE metrics on the test toy data.
