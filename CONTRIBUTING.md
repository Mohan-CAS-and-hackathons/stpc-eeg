# Contributing to the STPC Framework

First off, thank you for considering contributing! This project is an open-source research framework, and we welcome any contributions, from fixing typos to proposing novel deep learning architectures.

## How Can I Contribute?

### Reporting Bugs

If you find a bug, please open a new issue on our GitHub repository. Be sure to include:

- A clear and descriptive title.
- Steps to reproduce the bug.
- Any relevant error messages or logs.
- Your operating system and Python version.

### Proposing New Features or Experiments

We'd love to hear your ideas! Please open an issue with the "enhancement" or "research idea" label and provide:

- A clear description of the proposed feature or experiment.
- The motivation behind it (what problem does it solve?).
- (Optional) Any references to relevant papers or implementations.

### Submitting Code (Pull Requests)

We follow a standard Fork & Pull Request workflow.

1.  **Fork the repository** to your own GitHub account.
2.  **Clone your fork** to your local machine: `git clone https://github.com/YourUsername/ecg-denoiser-hackathon.git`
3.  **Set up your development environment:**
    ```bash
    cd ecg-denoiser-hackathon
    pip install -r requirements.txt
    ```
4.  **Create a new branch** for your changes: `git checkout -b feature/my-new-feature` or `bugfix/fix-that-bug`.
5.  **Make your changes.** Please adhere to our coding style.
6.  **Format your code.** We use the `black` code formatter to maintain a consistent style.
    ```bash
    pip install black
    black src/
    ```
7.  **Commit your changes** with a descriptive commit message.
8.  **Push your branch** to your fork: `git push origin feature/my-new-feature`.
9.  **Open a Pull Request** from your fork's branch to the `main` branch of the original repository. Please provide a clear description of the changes you've made.

We will review your PR as soon as possible. Thank you for your contribution!
