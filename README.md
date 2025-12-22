# 🚀 HuggingFace to ModelScope Migration Tool

A user-friendly Gradio application for migrating models and datasets from [HuggingFace](https://huggingface.co/) to [ModelScope](https://www.modelscope.cn/).

## ✨ Features

- **Easy Migration**: Migrate models and datasets with a simple web interface
- **Secure Authentication**: Supports token-based authentication for both platforms
- **Progress Tracking**: Real-time status updates during migration
- **Flexible Configuration**: Customize repository visibility, license, and metadata
- **Error Handling**: Comprehensive error messages and validation
- **Automatic Cleanup**: Temporary files are automatically cleaned up after migration

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

1. Run the application:
```bash
python app.py
```

2. Open your browser and navigate to the provided URL (usually `http://localhost:7860`)

3. Fill in the required information:
   - **HuggingFace Token**: Get from https://huggingface.co/settings/tokens
   - **ModelScope Token**: Get from https://www.modelscope.cn/my/myaccesstoken
   - **Source Repository**: The HuggingFace repository ID (e.g., `bert-base-uncased`)
   - **Destination Repository**: Your desired ModelScope repository ID (e.g., `username/my-model`)

4. Click "Start Migration" and wait for completion

## 📖 Usage Guide

### Getting Your Tokens

#### HuggingFace Token:
1. Go to https://huggingface.co/settings/tokens
2. Click "New token"
3. Give it a name and select appropriate permissions
4. Copy the token (starts with `hf_`)

#### ModelScope Token:
1. Go to https://www.modelscope.cn/my/myaccesstoken
2. Log in to your account
3. Copy your SDK token

### Migration Options

| Option | Description | Default |
|--------|-------------|---------|
| **Repository Type** | Choose between `model` or `dataset` | `model` |
| **Visibility** | Set repository as `public` or `private` | `public` |
| **License** | Choose from Apache 2.0, MIT, GPL-3.0, or Other | `apache-2.0` |
| **Chinese Name** | Optional Chinese name for the repository | None |

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

### "Authentication failed"
- Verify your tokens are correct and not expired
- Check that you're using the right token for each platform

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
