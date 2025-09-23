# STPC: A Physics-Aware AI Framework for Accessible Cardiac Diagnostics

[![Project Status: Complete](https://img.shields.io/badge/status-complete-green.svg)](https://github.com/Mohan-CAS-and-hackathons/ecg-denoiser-hackathon)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

This repository contains the complete source code, experimental data, and final research paper for the STPC project, a novel AI framework designed to make cardiac diagnostics reliable and accessible, even with noisy data from low-cost devices.

---

### Quick Links
**[🚀 View the Live Streamlit Demo](https://ecg-denoiser-hackathon-mohan.streamlit.app/)**
- **[📄 Read the Full Research Paper (PDF)](STPC_Research_Paper.pdf)**
- **[💻 Explore the Code](src)**

---

### Project Showcase

![Final Streamlit App Screenshot](assets/app_screenshot.png)
*The final web application demonstrating the end-to-end pipeline: a noisy ECG is uploaded, cleaned by the STPC model, and analyzed, resulting in a corrected and reliable diagnosis in seconds.*

---

### The Core Innovation: STPC Framework

The central contribution of this project is the **Spectral-Temporal Physiological Consistency (STPC)** framework, a new physics-informed method for training deep learning models on biomedical signals.

Traditional models often fail because they only match a signal's shape, leading to oversmoothing of critical features. The STPC framework teaches the AI the **physics of a real heartbeat** by forcing it to preserve three key properties simultaneously:

1.  **Amplitude Consistency:** The correct voltage and overall shape.
2.  **Temporal-Gradient Consistency:** A custom gradient loss that preserves the sharp, high-velocity spikes of a heartbeat (the QRS complex).
3.  **Spectral-Magnitude Consistency:** An FFT-based loss that ensures the denoised signal has a realistic frequency profile.

This results in a denoised signal that is not just clean, but **physiologically faithful** and trustworthy enough for downstream diagnostic tasks. Our research shows this approach leads to a measurable improvement in arrhythmia classification accuracy.

---

### How to Run Locally

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Mohan-CAS-and-hackathons/ecg-denoiser-hackathon.git
    cd ecg-denoiser-hackathon
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Run the Streamlit application:**
    ```bash
    streamlit run app.py
    ```

---
### Acknowledgements
This work was made possible by the open-source community and public datasets provided by PhysioNet.