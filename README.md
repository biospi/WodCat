# Early disease detection classifiers for wearables on domestic cat
This repository contains the source code that accompanies our paper "Early disease detection classifiers for wearables on domestic cat" - available at xxx

<div style="text-align:center">
  <img src="study_cat.jpg" alt="One of the cats that took part in the study." title="One of the cats that took part in the study." />
</div>

## How To Use

1) Clone the repository.
```bash
git clone https://github.com/biospi/WodCat.git
```
2) Change directory
```bash
cd WodCat/
```
3) Create python virtual environment 
```bash
python3 -m venv venv
```
4) Activate the environment
```bash
source venv/bin/activate
```
5) Install dependencies 
```bash
pip install --upgrade pip==21.3.1
make environment
```

## Dataset
Download dataset here xxx 
```bash
curl xxx
```

## Reproduce paper results
Run paper.py
```bash
python paper.py
```

## Create dataset of activity peaks

```bash
Usage: build_dataset.py [OPTIONS]

  Script which builds dataset ready for ml Args:

      data_dir: Activity data directory.

      out_dir: Output directory.

      bin: Activity bin size (activity count will be summed), 'T' for minutes
      and 'S' for seconds .

      w_size: Sample lengh (if bin is S, 60 give 60 seconds sample length).

      thresh: Top n highest values.

      n_peaks: Number of peaks in dataset.

      out_heatmap: Enables output of visualisation heatmaps.

      max_sample: Maximum number of samples per cats when using n_peaks > 1.

      n_job: Number of threads to use.

Options:
  --data-dir DIRECTORY            [required]
  --out-dir DIRECTORY             [required]
  --bin TEXT                      [default: S]
  --w-size INTEGER                [default: 10, 30, 60, 90]
  --threshs INTEGER               [default: 10, 100, 1000]
  --n-peaks INTEGER               [default: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
  --out-heatmap / --no-out-heatmap
                                  [default: no-out-heatmap]
  --max-sample INTEGER            [default: 10000]
  --n-job INTEGER                 [default: 6]
  --help                          Show this message and exit.
```

## Run the Machine learning pipeline

```bash
Usage: run_ml.py [OPTIONS]

  Thesis script runs the cats study Args:

      out_parent: Output directory     dataset_parent: Dataset directory

Options:
  --dataset-filepath FILE         [required]
  --out-dir DIRECTORY             [required]
  --clf TEXT                      [default: rbf]
  --preprocessing-steps TEXT      [default: QN]
  --meta-columns TEXT
  --cv TEXT                       [default: LeaveOneOut]
  --export-hpc-string / --no-export-hpc-string
                                  [default: no-export-hpc-string]
  --n-job INTEGER                 [default: 4]
  --help                          Show this message and exit.
```

## Collaborators
[![Bristol Veterinary School](http://www.bristol.ac.uk/media-library/protected/images/uob-logo-full-colour-largest-2.png)](http://www.bristol.ac.uk/vetscience/)