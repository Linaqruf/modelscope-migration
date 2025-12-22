---
license: apache-2.0
title: Modelscope Migration
sdk: gradio
emoji: 🚀
colorFrom: red
colorTo: blue
short_description: Migrate your models to modelscope
sdk_version: 6.2.0
---

# 🚀 HuggingFace to ModelScope Migration Tool

A user-friendly Gradio application for migrating models and datasets from [HuggingFace](https://huggingface.co/) to [ModelScope](https://www.modelscope.cn/).

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
git clone <repository-url>
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

#### ModelScope Token (CRITICAL):
1. **Domain**: You MUST use tokens from **[modelscope.cn](https://www.modelscope.cn/)** (Chinese site). The international site (`modelscope.ai`) is not currently supported by the SDK.
2. **Type**: Use an **SDK Token** from https://www.modelscope.cn/my/myaccesstoken.
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

### Error Handling

The tool provides clear error messages for:
- Invalid tokens
- Repository not found
- Permission errors
- Network issues
- Disk space problems

## ⚠️ Important Notes

- **Disk Space**: Ensure you have enough disk space for temporary storage of the repository
- **Large Models**: Migration time depends on repository size and network speed
- **Permissions**: Your tokens must have appropriate read/write permissions
- **Private Repos**: To migrate private repositories, ensure your HuggingFace token has access
- **Repository Creation**: The tool automatically creates the ModelScope repository if it doesn't exist

## 🔒 Security

- Tokens are handled securely and not logged
- Tokens are entered as password fields (hidden input)
- Temporary files are cleaned up after migration
- No data is stored permanently on the server

## 🐛 Troubleshooting

### "Authentication failed" (400 Client Error)
- **Whitespace**: Ensure there are no leading or trailing spaces in your tokens.
- **Wrong Domain**: Verify you are using a token from **modelscope.cn**, not modelscope.ai.
- **Wrong Token**: Ensure you are using the **SDK Token** (starts with `ms-`), not a Git token.

### "Repository format error"
- **Namespace**: ModelScope repository IDs *must* be in the format `username/repo-name`.

### "Download failed"
- Ensure the HuggingFace repository exists and is accessible
- Check your internet connection
- Verify you have permission to access the repository

### "Upload failed"
- Verify your ModelScope token has write permissions
- Check that the destination repository name is valid
- Ensure you're not exceeding ModelScope's quotas

### "Disk space error"
- Free up disk space on your system
- Large models require significant temporary storage

## 📚 API Documentation

### HuggingFace Hub
- Documentation: https://huggingface.co/docs/huggingface_hub
- PyPI: https://pypi.org/project/huggingface-hub/

### ModelScope
- GitHub: https://github.com/modelscope/modelscope
- PyPI: https://pypi.org/project/modelscope/

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📜 Changelog

### [2025-12-22]
- **Fixed**: Reliable private repository creation. Previously, if the initial creation failed, the tool would fallback to `upload_folder` which defaults to public visibility. Now, repository creation is explicitly handled with the user-selected visibility before any upload starts.

## 📄 License

This project is open source and available under the Apache 2.0 License.

## 🔗 Resources

- [HuggingFace Hub](https://huggingface.co/)
- [ModelScope Platform](https://www.modelscope.cn/)
- [Gradio Documentation](https://gradio.app/docs/)

## 💡 Tips

1. **Test with small models first**: Before migrating large models, test with a smaller repository
2. **Check licenses**: Ensure you have the right to migrate and redistribute the model/dataset
3. **Preserve metadata**: The tool preserves all files, but review the ModelScope repository after migration
4. **Network stability**: For large repositories, ensure a stable internet connection

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review the error message in the status output
3. Verify your tokens and repository IDs
4. Check the official documentation for HuggingFace and ModelScope

---

Made with ❤️ using [Gradio](https://gradio.app/), [HuggingFace Hub](https://huggingface.co/), and [ModelScope](https://www.modelscope.cn/)
