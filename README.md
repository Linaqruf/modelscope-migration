
# 🚀 HuggingFace to ModelScope Migration Tool

A user-friendly Gradio application for migrating models and datasets from [HuggingFace](https://huggingface.co/) to [ModelScope](https://www.modelscope.ai/).

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/Linaqruf/modelscope-migration)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Linaqruf/modelscope-migration/blob/main/notebook.ipynb)

## ✨ Features

- **Easy Migration**: Migrate models and datasets with a simple web interface or a headless CLI.
- **Real-Time Console Logs**: See actual library logs and progress bars directly in your browser.
- **Smart Progress Tracking**: Visual progress bar and stable, in-place updates for multiple files.
- **CLI Mode**: Fully automated migrations from the terminal without a browser.
- **Secure Authentication**: Robust token validation and domain guidance for ModelScope.
- **Automatic Cleanup**: Temporary files are automatically removed after every migration.
- **Flexible Configuration**: Customize visibility, licenses, and metadata.

## 📋 Prerequisites

- Python 3.8 or higher
- Internet connection
- HuggingFace account and access token
- ModelScope account and SDK token

## 🔧 Installation

1. Clone this repository:
```bash
git clone https://github.com/Linaqruf/modelscope-migration
cd modelscope-migration
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install gradio huggingface-hub modelscope
```

## 🚀 Quick Start

### Web Interface
1. Run the application:
```bash
python app.py
```

2. Open your browser to `http://localhost:7860`.

### CLI Mode (Headless)
For automation or remote servers, use the CLI:
```bash
python app.py --cli \
    --hf-token "your_hf_token" \
    --ms-token "your_ms_token" \
    --hf-repo "username/source-model" \
    --ms-repo "username/dest-model"
```

## 📖 Usage Guide

### Getting Your Tokens

#### HuggingFace Token:
1. Go to https://huggingface.co/settings/tokens
2. Copy your token (starts with `hf_`).

#### ModelScope Token:
1. **Domain**: This tool is configured to use the international site **[modelscope.ai](https://www.modelscope.ai/)**.
2. **Type**: Use an **SDK Token** from https://www.modelscope.ai/my/myaccesstoken.
3. **Prefix**: The correct token usually starts with `ms-`.

### Advanced CLI Options

| Argument | Description | Default |
|----------|-------------|---------|
| `--host` | Host for Gradio app | `127.0.0.1` |
| `--port` | Port for Gradio app | `7860` |
| `--share` | Create public Gradio link | `False` |
| `--type` | `model` or `dataset` | `model` |
| `--visibility` | `public` or `private` | `public` |
| `--license` | Repo license (e.g., `mit`) | `apache-2.0` |
| `--chinese-name`| Optional Chinese name | None |

### Example Repository IDs

**HuggingFace:**
- `bert-base-uncased` (official model)
- `username/my-custom-model` (user model)
- `datasets/squad` (dataset)

**ModelScope:**
- `username/repo-name` (always requires username prefix)

## 🛠️ How It Works

The migration process follows these steps:

1. **Authentication**: Validates tokens for both HuggingFace and ModelScope
2. **Download**: Uses `snapshot_download` to download the entire repository from HuggingFace
3. **Upload**: Creates repository on ModelScope (if needed) and uploads all files
4. **Cleanup**: Removes temporary files from local storage

## 📝 Features in Detail

### Model Migration

Migrates complete model repositories including:
- Model weights and configuration files
- Tokenizer files
- README and documentation
- Any additional files in the repository

### Dataset Migration

Migrates dataset repositories including:
- Dataset files (all formats)
- Dataset cards and metadata
- Scripts and configuration files

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📜 Changelog

### [2025-12-22]
- **Fixed**: Reliable private repository creation. Previously, if the initial creation failed, the tool would fallback to `upload_folder` which defaults to public visibility. Now, repository creation is explicitly handled with the user-selected visibility before any upload starts.
- **Fixed**: License handling for repository creation. The ModelScope API now correctly handles cases where no license is specified (e.g., "other" license type). Previously, passing `null` for the license field would result in a 400 Bad Request error. The license parameter is now only included in API calls when a valid license is selected.

## 📄 License

This project is open source and available under the Apache 2.0 License.

## 🔗 Resources

- [HuggingFace Hub](https://huggingface.co/)
- [ModelScope Platform](https://www.modelscope.ai/)
- [Gradio Documentation](https://gradio.app/docs/)

---

Made with ❤️ using [Gradio](https://gradio.app/), [HuggingFace Hub](https://huggingface.co/), and [ModelScope](https://www.modelscope.ai/)
