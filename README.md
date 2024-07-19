# eeg_ms_classification
- this code is full-flow of multiple sclerosis classification with 2 classes: intact and decrease cognition.

- import data (.set file) to folder Data in this github.

- run full-flow.py to export .csv file (include 95 features and 60 patients) and run classification-eeg.py to classify intact and decrease cognition.

- preview the interface for predicting MS disease using EEG data: ![image](https://github.com/user-attachments/assets/cb043fab-252e-4b53-b8f5-1e8456831edd)


## Features

### One Subject
#### EEG information extraction
Extracts relevant information from raw EEG data files.
- **Input**: raw EEG data files [(.set+.fdt), ...]
- **Output**: times, sampling frequency, label channels, ...
```
def eegExt()
    return
```

#### Transform to Frequency Spectrum
Transforms time-domain EEG signals to the frequency spectrum using Power Spectral Density (PSD).
- **Input**: PSD transform method, times, window, sample, overlap, ...
- **Output**: power spectrum density vectors for 1 channel
```
def psdTrans()
    
    return
```

#### Power Spectral feature extraction (for 1 subject)
Extracts spectral features (periodic & aperiodic components) from PSD vectors for a single subject.
- **Input**: PSD vectors (n channels), fooof settings, ...  
- **Output**: feature vectors (5*n features)
```
def psdExt()
    
    return
```


### Subjects Group
