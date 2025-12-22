"""
Example script demonstrating programmatic usage of the HuggingFace to ModelScope Migration Tool.

This example shows how to use the MigrationTool class directly without the Gradio UI.
"""

import os
from app import MigrationTool


def main():
    """Example migration workflow."""

    # Initialize the migration tool
    tool = MigrationTool()

    # Configuration - Replace these with your actual values
    # IMPORTANT: Never hardcode tokens in production code!
    # Use environment variables or secure credential management instead.
    HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN", "")
    MS_TOKEN = os.getenv("MODELSCOPE_TOKEN", "")

    # Source repository on HuggingFace
    HF_REPO_ID = "sentence-transformers/all-MiniLM-L6-v2"  # Example small model

    # Destination repository on ModelScope
    MS_REPO_ID = "your-username/all-MiniLM-L6-v2"  # Replace with your repo

    # Repository configuration
    REPO_TYPE = "model"  # or "dataset"
    VISIBILITY = "public"  # or "private"
    LICENSE = "apache-2.0"  # apache-2.0, mit, gpl-3.0, other
    CHINESE_NAME = "小型句子转换模型"  # Optional

    # Validate tokens
    if not HF_TOKEN or not MS_TOKEN:
        print("Error: Please set HUGGINGFACE_TOKEN and MODELSCOPE_TOKEN environment variables")
        print("\nExample:")
        print("  export HUGGINGFACE_TOKEN='hf_...'")
        print("  export MODELSCOPE_TOKEN='your-token'")
        print("  python example_usage.py")
        return

    print("=" * 60)
    print("HuggingFace to ModelScope Migration Tool - Example Usage")
    print("=" * 60)
    print()
    print(f"Source: {HF_REPO_ID}")
    print(f"Destination: {MS_REPO_ID}")
    print(f"Type: {REPO_TYPE}")
    print(f"Visibility: {VISIBILITY}")
    print()

    # Perform the migration
    result = tool.migrate(
        hf_token=HF_TOKEN,
        ms_token=MS_TOKEN,
        hf_repo_id=HF_REPO_ID,
        ms_repo_id=MS_REPO_ID,
        repo_type=REPO_TYPE,
        visibility=VISIBILITY,
        license_type=LICENSE,
        chinese_name=CHINESE_NAME
    )

    # Print the result
    print(result)
    print()
    print("=" * 60)


if __name__ == "__main__":
    main()
