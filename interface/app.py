import streamlit as st
import os
import mne
import numpy as np
from scipy.signal import welch, hamming
from fooof import FOOOF
import pickle

# Hàm tải mô hình từ file
def load_model(file_path):
    with open(file_path, 'rb') as file:
        model = pickle.load(file)
    return model

# Tải mô hình từ file
model = load_model("model.pkl")

# Hàm xử lý file .set và trích xuất các đặc trưng
def process_file(set_file_path, fdt_file_path):
    # Kiểm tra xem file .fdt có tồn tại không
    if not os.path.exists(fdt_file_path):
        st.error(f"Missing corresponding .fdt file for {set_file_path}")
        return None
    
    # Đọc file .set
    raw = mne.io.read_raw_eeglab(set_file_path, preload=True)
    channel_name = raw.info['ch_names']
    all_features = []
    channel_name = raw.info['ch_names']
    channel_index = []
    channel_data = [] 
    channel_time = []
    frequencies = []
    psd = []
    all_features = []
    # Trích xuất đặc trưng cho mỗi kênh
    for i in range(19):
        index = raw.ch_names.index(channel_name[i])
        data = raw[index, :][0] * 1e6
        time = raw[index, :][1]

        channel_index.append(index)
        channel_data.append(data)
        channel_time.append(time)
        freqs, power = welch(channel_data[i], fs=raw.info['sfreq'], nperseg=512, window=hamming(512))
        frequencies.append(freqs)
        psd.append(power)
        
        fm_periodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=1, min_peak_height=0.01, 
                   peak_threshold=-5, aperiodic_mode='fixed')
        fm_aperiodic = FOOOF(peak_width_limits=[0.05, 20], max_n_peaks=99999999999, min_peak_height=0.01, 
                   peak_threshold=-10, aperiodic_mode='fixed')
        freq_range_periodic = [4, 16]
        freq_range_aperiodic = [0.5, 40]
        fm_periodic.report(frequencies[i], psd[i][0], freq_range_periodic)
        fm_aperiodic.report(frequencies[i], psd[i][0], freq_range_aperiodic)
        aperiodic_params = fm_aperiodic.get_params('aperiodic_params')
        peak_params = fm_periodic.get_params('peak_params')
        peak_params = peak_params.flatten()
        peak_params = peak_params[:2]
        if np.isnan(peak_params).any():
            peak_params = np.array([0, 0, 0])
        features = np.concatenate((peak_params, aperiodic_params))
        all_features.extend(features)
    return all_features 

# Giao diện Streamlit
st.title("Predicting multiple sclerosis using EEG data")

# Tải file .set và .fdt cùng một lúc
uploaded_files = st.file_uploader("Upload .set and .fdt files (same name)", type=["set", "fdt"], accept_multiple_files=True)

# Lưu file và xử lý
if uploaded_files:
    set_file = None
    fdt_file = None
    
    # Phân loại và lưu file
    for uploaded_file in uploaded_files:
        if uploaded_file.name.endswith(".set"):
            set_file = uploaded_file
        elif uploaded_file.name.endswith(".fdt"):
            fdt_file = uploaded_file
    
    if set_file and fdt_file:
        # Lưu file tạm thời
        set_file_path = "temp.set"
        fdt_file_path = "temp.fdt"
        
        with open(set_file_path, "wb") as f:
            f.write(set_file.getbuffer())
        
        with open(fdt_file_path, "wb") as f:
            f.write(fdt_file.getbuffer())
        
        st.write("The files have been uploaded successfully. Processing...")
        
        # Xử lý file và trích xuất đặc trưng
        features = process_file(set_file_path, fdt_file_path)
        
        if features is not None:
            if len(features) == 76:
                input_data = np.array(features).reshape(1, -1)
                prediction = model.predict(input_data)
                if prediction[0] == 1:
                    st.success('This person shows signs of cognitive decline')
                else:
                    st.success('This person showed no signs of cognitive impairment')
            else:
                st.error("Characteristic processing failure. Please check the .set file.")
    else:
        st.error("Please upload both .set and .fdt files at the same time!")
