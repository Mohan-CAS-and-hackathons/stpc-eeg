import streamlit as st
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
import plotly.graph_objects as go
from PIL import Image
import torch
from collections import Counter
import os, sys, pathlib, base64

# --- System Setup ---
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# --- UPDATED IMPORTS ---
from src.stpc.inference import denoise_ecg_signal, DEVICE
from src.stpc.utils.ecg_utils import TARGET_FS
from src.stpc.model import UNet1D, ECGClassifier
# ---

st.set_page_config(
    page_title="STPC: AI for Reliable Cardiac Diagnostics",
    page_icon="❤️",
    layout="wide",
)

INT_TO_AAMI_CLASS = {0: 'Normal (N)', 1: 'Supraventricular (S)', 2: 'Ventricular (V)', 3: 'Fusion (F)', 4: 'Unknown (Q)'}

@st.cache_resource
def get_models():
    denoiser, classifier = None, None
    try:
        denoiser = UNet1D(in_channels=1, out_channels=1)
        denoiser.load_state_dict(torch.load('models/denoiser_stpc_full.pth', map_location=DEVICE))
        denoiser.to(DEVICE); denoiser.eval()
        
        classifier = ECGClassifier(num_classes=5)
        classifier.load_state_dict(torch.load('models/ecg_classifier_model.pth', map_location=DEVICE))
        classifier.to(DEVICE); classifier.eval()
        print("✅ Models loaded successfully.")
    except Exception as e:
        st.error(f"Fatal Error loading models: {e}.")
    return denoiser, classifier

denoiser_model, classifier_model = get_models()

# --- Analysis & Plotting Functions (No changes needed here) ---
def analyze_ecg(signal, fs):
    try:
        peaks, _ = find_peaks(signal, height=np.mean(signal) + 0.75 * np.std(signal), distance=fs * 0.4)
        if len(peaks) < 2: return "N/A", "N/A", peaks
        rr_intervals = np.diff(peaks) / fs
        heart_rate = 60 / np.mean(rr_intervals)
        cov = np.std(rr_intervals) / np.mean(rr_intervals)
        rhythm_status = "Regular" if cov < 0.15 else "Irregular"
        return f"{heart_rate:.1f} bpm", rhythm_status, peaks
    except Exception: return "N/A", "N/A", []

def classify_beats(signal, peaks, classifier, fs, window_size=128):
    if classifier is None or len(peaks) == 0: return {}
    predictions = []
    for p in peaks:
        start, end = p - window_size//2, p + window_size//2
        if start >= 0 and end < len(signal):
            beat_window = signal[start:end].astype(np.float32)
            with torch.no_grad():
                tensor_in = torch.from_numpy(beat_window).unsqueeze(0).unsqueeze(0).to(DEVICE)
                pred_label = torch.argmax(classifier(tensor_in), dim=1).item()
                predictions.append(INT_TO_AAMI_CLASS.get(pred_label, 'Unknown'))
    return Counter(predictions)

def create_ecg_plot(signal, peaks, title, fs):
    time_axis = np.arange(len(signal)) / fs
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time_axis, y=signal, mode='lines', name='ECG', line=dict(color='royalblue')))
    if len(peaks) > 0:
        fig.add_trace(go.Scatter(x=time_axis[peaks], y=signal[peaks], mode='markers', name='R-peaks', marker=dict(color='red', size=8, symbol='x')))
    fig.update_layout(title_text=title, xaxis_title="Time (s)", yaxis_title="Amplitude", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig



# --- Main App UI ---
st.title("❤️ STPC: From Noise to Diagnosis")
st.subheader("An AI Framework for Reliable Cardiac Monitoring, Anywhere.")

tab1, tab2, tab3, tab4 = st.tabs(["**🚀 Live Demo**", "**🔬 The Problem & My Solution**", "**📈 Validation Results**", "**💻 The Technology**"])

# --- TAB 1: Live Demo ---
with tab1:
    st.header("Transform a Noisy ECG into a Confident Diagnosis")

    with st.sidebar:
        st.image("assets/header_image.png", use_column_width='auto')
        st.title("Get Started")
        st.markdown(
            """
            1.  **Download the sample file** to see the app in action with a challenging, real-world signal.
            2.  **Upload the sample file** (or your own single-column CSV).
            3.  Review the AI-powered denoising and corrected automated diagnosis.
            """
        )
        # Robust check for the sample file
        sample_file_path = "samples/sample_noisy_ecg.csv"
        if os.path.exists(sample_file_path):
            with open(sample_file_path, "rb") as file:
                st.download_button(label="Download Sample ECG (.csv)", data=file, file_name="sample_noisy_ecg.csv", mime="text/csv")
        else:
            st.error("Sample file not found. Please run `python -m src.create_sample` from your terminal.")
        st.markdown("---")
        uploaded_file = st.file_uploader("Upload your noisy ECG file", type=["csv", "txt"])
    
    if uploaded_file is not None:
        try:
            noisy_signal_np = pd.read_csv(uploaded_file, header=None)[0].to_numpy()
            
            # Pad or truncate the signal to the model's expected input size (2048)
            if len(noisy_signal_np) != 2048:
                st.warning(f"⚠️ Input signal has {len(noisy_signal_np)} samples, but the model expects 2048. Signal will be padded/truncated.", icon="⚠️")
                final_signal = np.zeros(2048)
                len_to_copy = min(len(noisy_signal_np), 2048)
                final_signal[:len_to_copy] = noisy_signal_np[:len_to_copy]
                noisy_signal_np = final_signal

            st.subheader("Processing...")
            
            with st.spinner('AI is cleaning the signal using the STPC framework...'):
                denoised_signal_np = denoise_ecg_signal(noisy_signal_np, denoiser_model)
            
            hr_noisy, rhythm_noisy, peaks_noisy = analyze_ecg(noisy_signal_np, TARGET_FS)
            hr_denoised, rhythm_denoised, peaks_denoised = analyze_ecg(denoised_signal_np, TARGET_FS)
            
            with st.spinner('AI is classifying individual heartbeats...'):
                beat_counts = classify_beats(denoised_signal_np, peaks_denoised, classifier_model, TARGET_FS)

            st.success("Processing Complete!", icon="✅")
            
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Original Noisy Signal")
                st.metric("Heart Rate", hr_noisy)
                st.metric("Rhythm", rhythm_noisy)
                st.plotly_chart(create_ecg_plot(noisy_signal_np, peaks_noisy, "Original Noisy ECG", TARGET_FS), use_container_width=True, key=1)
            
            with col2:
                st.subheader("AI Denoised & Analyzed Signal")
                st.metric("Heart Rate", hr_denoised)
                st.metric("Rhythm", rhythm_denoised)
                if beat_counts:
                    st.markdown("**Detected Beat Types:**")
                    st.json(dict(beat_counts))
                st.plotly_chart(create_ecg_plot(denoised_signal_np, peaks_denoised, "AI Denoised ECG (STPC Model)", TARGET_FS), use_container_width=True, key=2)
        except Exception as e:
            st.error(f"An error occurred during processing. Please ensure the uploaded file is a single-column CSV. Error: {e}")
    else:
        st.info("⬆️ Upload a noisy ECG file or use the sample to begin analysis.")

# --- TAB 2: The Problem & My Solution ---
with tab2:
    st.header("The Problem: Data We Can't Trust")
    st.markdown(
        """
        AI-driven healthcare promises a revolution in diagnostics, but it has a critical weakness: **it's only as good as the data it's fed.** In many real-world settings, crucial data is corrupted by noise, leading to diagnostic errors.
        
        This is especially true for cardiac care:
        - **Wearable Technology (e.g., Smartwatches):** Noise from daily activities can mask conditions like **Atrial Fibrillation**, a leading cause of stroke.
        - **Rural & Emergency Medicine:** In clinics with portable ECGs or in ambulances, patient movement can render a signal unreadable, delaying diagnosis of a **heart attack** when every minute counts.
        - **Intensive Care Units (ICUs):** Constant interference can cause **alarm fatigue**, where staff begin to ignore alerts, potentially missing a real crisis.
        """
    )

    st.header("My Solution: The STPC Framework")
    st.markdown(
        """
        To solve this "garbage-in, garbage-out" problem, I developed a novel AI training framework: **STPC (Spectral-Temporal Physiological Consistency)**. Instead of just teaching an AI to remove noise, I taught it the **physics of a real heartbeat.**

        My framework forces a **1D U-Net** model to preserve three essential properties of a real ECG:
        - **1. Amplitude Consistency:** The basic voltage levels must be correct.
        - **2. Temporal-Gradient Consistency:** A custom **gradient loss** preserves the sharp, high-velocity spikes of a heartbeat, preventing dangerous oversmoothing.
        - **3. Spectral-Magnitude Consistency:** An **FFT-based loss** ensures the output has a realistic frequency profile, matching the harmonic signature of a true ECG.

        This approach produces a signal that is not just clean, but **trustworthy and physiologically faithful**, bridging the gap between noisy data and confident clinical decisions.
        """
    )

# --- TAB 3 (REVISED): Validation Results & Generalization ---
with tab3:
    st.header("Proof of Impact Part 1: Improving ECG Diagnosis")
    st.markdown(
        """
        To prove that STPC works, I conducted a rigorous ablation study on ECG data. I trained different denoiser models and tested them on an unseen patient record to measure how their denoising affected the accuracy of a separate diagnostic AI.
        """
    )
    st.subheader("Quantitative Results: STPC Dramatically Improves Diagnostic Accuracy")
    st.markdown("The F1-score is a key metric that balances precision and recall. A higher score is better. The table below shows a comparison of the diagnostic AI's performance on the raw noisy signal versus the signal cleaned by my **best STPC model**.")
    
    best_results_data = {
        'Beat Type': ['Normal (N)', 'Supraventricular (S)', 'Ventricular (V)', '**Overall Accuracy**'],
        'Performance on Noisy Signal (F1-Score)': ['0.97', '0.28', '0.55', '90.2%'],
        'Performance on STPC Denoised Signal (F1-Score)': ['**0.98**', '**0.52**', '**0.74**', '**96.3%**']
    }
    st.table(pd.DataFrame(best_results_data).set_index('Beat Type'))
    st.success(
        """
        **Key Takeaway:** By cleaning the signal first, my STPC framework increased the F1-score for detecting **Supraventricular beats by 85%** and critical **Ventricular beats by 35%**.
        """
    )
    
    st.subheader("Visual Comparison: From Chaos to Clarity")
    st.markdown("The confusion matrices below visually confirm the improvement. A perfect matrix has a bright diagonal line. Notice how noise completely confuses the classifier (left), while the signal cleaned by my STPC model (right) allows for a much more accurate diagnosis.")
    
    # Use string paths directly
    if os.path.exists('results/final_cm_noisy.png') and os.path.exists('results/final_cm_stpc_full_denoised.png'):
        col1, col2 = st.columns(2)
        with col1:
            st.image('results/final_cm_noisy.png', caption="Classifier Performance on RAW NOISY Data")
        with col2:
            st.image('results/final_cm_stpc_full_denoised.png', caption="Classifier Performance on STPC DENOISED Data")
    else:
        st.warning("ECG validation images not found. Please run the validation scripts to generate all result files.")

    # --- EXPANDED GENERALIZATION SECTION ---
    st.markdown("---")
    st.header("Proof of Impact Part 2: Generalizing to Brain Signals (EEG)")
    st.markdown(
        """
        A successful model for ECGs is great, but a truly powerful framework should be versatile. To prove that STPC wasn't just a fluke, we tested it on a completely different and more complex challenge: **denoising EEG (brain wave) signals from a patient during an epileptic seizure.**
        """
    )
    
    st.subheader("Finding 1: Preserving Critical Diagnostic Features")
    st.markdown("The onset of a seizure is a sharp, chaotic spike. A simple denoiser might oversmooth this event. The plot below shows that the STPC model (green) perfectly reconstructs the seizure spike's shape and underlying dynamics (gradient), while a basic model (red) fails.")
    
    if os.path.exists("results/eeg_gradient_preservation_plot.png"):
        st.image("results/eeg_gradient_preservation_plot.png", caption="STPC (green) preserves the sharp seizure spike's shape and gradient.")
    else:
        st.warning("EEG gradient plot not found. Please run the EEG experiments to generate it.")
    
    st.subheader("Finding 2: The Power of Unsupervised Learning")
    st.markdown(
        """
        Our most significant result came from a self-supervised learning experiment. We gave the STPC-regularized model a mix of seizure and non-seizure EEG data **without any labels** and tasked it with reconstructing masked segments of the signal. The result was remarkable: by learning the *physical rules* of EEG signals, the model spontaneously learned to tell the difference between healthy and pathological brain activity.
        """
    )

    if os.path.exists("results/phase3_embedding_comparison.png"):
        st.image("results/phase3_embedding_comparison.png", caption="UMAP Projection: Without labels, the AI learned to separate healthy (purple) and seizure (yellow) brain states.")
    else:
        st.warning("Phase 3 embedding plot not found. Please run the EEG experiments to generate it.")

    st.subheader("Finding 3: Maintaining Spatio-Temporal Plausibility")
    st.markdown("This is possible because the spatial component of STPC forces the AI to respect the physics of how signals spread across the brain. The animation below shows the STPC model (far right) faithfully reconstructing the complex topography of a seizure, while a basic model (center-right) collapses into a non-physiological mess.")

    def embed_mp4_loop_inline(mp4_path, width=600, max_inline_mb=15):
        """
        Embed mp4 as a base64 data URI so it autoplays/loops/mutes in Streamlit.
        If file is larger than max_inline_mb, fall back to st.video() to avoid huge pages.
        """
        p = pathlib.Path(mp4_path)
        if not p.exists():
            st.warning(f"File not found: {mp4_path}")
            return

        size_mb = p.stat().st_size / (1024*1024)
        if size_mb > max_inline_mb:
            st.warning(f"Video is {size_mb:.1f} MB — too large to embed inline. Using st.video() fallback.")
            st.video(str(p))
            return

        data = p.read_bytes()
        b64 = base64.b64encode(data).decode("ascii")
        html = f'''
        <video autoplay loop muted playsinline width="{width}">
        <source src="data:video/mp4;base64,{b64}" type="video/mp4">
        Your browser does not support the video tag.
        </video>
        '''
        # height roughly: 16:9 aspect; adjust if your video is different
        height = int(width * 9 / 16) + 30
        st.components.v1.html(html, height=height, scrolling=False)

    # Pass the GIF path directly as a string to enable animation
    mp4_path = "results/phase1_spatial_comparison.mp4"

    if os.path.exists("results/phase1_spatial_comparison.mp4"):
        # st.image("results/phase1_spatial_comparison.gif", caption="Comparison of scalp topographies. STPC (right) preserves the physical patterns of the seizure.")
        # st.video("results/phase1_spatial_comparison.mp4")
        # video_path = "results/phase1_spatial_comparison.mp4"
        # st.markdown(
        #     f"""
        #     <video autoplay loop muted playsinline width="600">
        #         <source src="{video_path}" type="video/mp4">
        #     </video>
        #     """,
        #     unsafe_allow_html=True,
        # )
        # gif_path = "results/phase1_spatial_comparison.gif"

        if os.path.exists(mp4_path):
            embed_mp4_loop_inline(mp4_path, width=600)

    else:
        st.warning("Phase 1 comparison GIF not found. Please run the EEG experiments to generate it.")

# --- TAB 4: The Technology ---
with tab4:
    st.header("A Modern, Open-Source Stack")
    st.markdown(
        """
        I built this project entirely with free, open-source tools to ensure its accessibility and reproducibility.

        - **AI & Machine Learning:** Python, PyTorch, Scikit-learn, NumPy, SciPy
        - **Data Source:** PhysioNet (MIT-BIH Arrhythmia & Noise Stress Test Databases)
        - **Web Application:** Streamlit
        - **Training & Experiments:** Google Colab (leveraging free T4 GPUs)

        The complete source code, including the training notebooks and the full research paper detailing the STPC framework, is available on GitHub.
        """
    )
    st.link_button("View on GitHub", "https://github.com/Mohan-CAS-and-hackathons/ecg-denoiser-hackathon")
    st.link_button("Read the Full Research Paper", "https://github.com/Mohan-CAS-and-hackathons/stpc-eeg/blob/main/manuscript/main.pdf")