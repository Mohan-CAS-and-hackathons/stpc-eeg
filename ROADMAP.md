# STPC Framework Roadmap

This document outlines the development and research goals for the STPC framework. Our vision is to build a robust, open-source toolkit for trustworthy biomedical AI.

## Short-Term Goals (Next 0-3 Months)

-   [ ] **Finalize and Publish Initial Research Papers:**
    -   [ ] Complete the manuscript for the core STPC framework (ECG + EEG generalization).
    -   [ ] Write and submit a paper on the self-supervised "Brain State" discovery findings.
-   [ ] **Onboard First Wave of Contributors:**
    -   [ ] Successfully guide new contributors through their first "Good First Issues".
    -   [ ] Validate the new refactored codebase by having contributors replicate key experimental results.
-   [ ] **Enhance Developer Experience & Documentation:**
    -   [ ] Create a comprehensive suite of tutorials in the `tutorials/` directory.
    -   [ ] Implement a configuration system (e.g., YAML files) for `experiments/run_training.py` to make experiments more manageable.
    -   [ ] Add robust unit and integration tests for the core `stpc` library functions.

## Mid-Term Goals (Next 3-9 Months)

-   [ ] **Research & Implement Advanced Self-Supervised Learning:**
    -   [ ] Move beyond masked reconstruction to state-of-the-art **Contrastive Learning** methods (e.g., SimCLR, MoCo) adapted for time-series data. The goal is to learn even richer and more discriminative representations.
-   [ ] **Expand Dataset Validation:**
    -   [ ] Test and validate the STPC framework on at least one new, large-scale public dataset (e.g., the TUH EEG Corpus) to further prove its generalization capabilities.
-   [ ] **Explore Alternative Spatial Modeling:**
    -   [ ] Implement a proof-of-concept Graph Convolutional Network (GCN) layer to learn spatial relationships in EEG data directly, as an alternative to the fixed Laplacian loss.

## Long-Term Vision (9+ Months)

-   **Towards a Generative "Brain-GPT":** Our ultimate goal is to move beyond discriminative tasks and representation learning into the realm of generative modeling. We aim to build a foundational model capable of:
    -   **Predicting** future neural activity based on past signals.
    -   **Synthesizing** physiologically realistic EEG/ECG data.
    -   Performing **in-silico experiments** by simulating neural responses to hypothetical stimuli.
-   **Clinical & Real-World Integration:** Explore pathways for packaging the models and framework into a tool that could be used for research in clinical settings or integrated with wearable device platforms.

We welcome community input on this roadmap! If you have ideas, please open an issue to discuss them.