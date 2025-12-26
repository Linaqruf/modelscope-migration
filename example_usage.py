"""
Example script demonstrating programmatic usage of the HuggingFace to ModelScope Migration Tool.

This module provides a programmatic interface for migrating repositories from HuggingFace to ModelScope
without using the Gradio UI. It demonstrates how to use the MigrationTool class directly and handle
authentication tokens, repository information, and migration parameters.

The module includes functions for performing migrations with automatic detection of repository
visibility and license information when not explicitly provided.
"""

import os
from typing import Optional, Literal

from huggingface_hub import HfApi

from app import MigrationTool


def migrate(hf_repo_id: str, repo_type: str = 'model', ms_repo_id: Optional[str] = None,
            visibility: Optional[Literal['public', 'private']] = None, licence_type: Optional[str] = None,
            chinese_name: Optional[str] = None) -> None:
    """
    Migrate a repository from HuggingFace to ModelScope.

    This function performs a complete migration of a repository from HuggingFace to ModelScope,
    including automatic detection of repository visibility and license information when not
    explicitly provided. It handles authentication tokens from environment variables and
    provides detailed progress information during the migration process.

    :param hf_repo_id: The HuggingFace repository ID to migrate from (e.g., 'username/repo-name').
    :type hf_repo_id: str
    :param repo_type: The type of repository to migrate. Defaults to 'model'.
    :type repo_type: str
    :param ms_repo_id: The ModelScope repository ID to migrate to. If None, uses the same ID as hf_repo_id.
    :type ms_repo_id: Optional[str]
    :param visibility: The visibility setting for the ModelScope repository ('public' or 'private').
                      If None, automatically determined based on the source repository.
    :type visibility: Optional[Literal['public', 'private']]
    :param licence_type: The license type for the repository. If None, automatically detected from
                        the source repository tags.
    :type licence_type: Optional[str]
    :param chinese_name: Optional Chinese name for the repository.
    :type chinese_name: Optional[str]

    :return: None
    :rtype: None

    :raises SystemExit: If required authentication tokens are not found in environment variables.
    """
    ms_repo_id = ms_repo_id or hf_repo_id

    tool = MigrationTool()
    hf_token = os.getenv("HUGGINGFACE_TOKEN", "") or os.getenv("HF_TOKEN", "")
    ms_token = os.getenv("MODELSCOPE_TOKEN", "") or os.getenv("MS_TOKEN", "")

    hf_client = HfApi(token=hf_token)
    hf_repo_info = hf_client.repo_info(repo_id=hf_repo_id, repo_type=repo_type)
    if visibility is None:  # auto selection for visibility when None is given
        if not hf_repo_info.private and not hf_repo_info.gated:
            visibility = 'public'
        else:
            visibility = 'private'

    if licence_type is None:  # auto find license from original huggingface repository
        for tag in hf_repo_info.tags:
            if tag.startswith('license:'):
                licence_type = tag[len('license:'):]
                break
        if licence_type is None:
            licence_type = 'other'

    if not hf_token or not ms_token:
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
    print(f"Source: {hf_repo_id}")
    print(f"Destination: {ms_repo_id}")
    print(f"Type: {repo_type}")
    print(f"Visibility: {visibility}")
    print(f"License: {licence_type}")
    print()

    # Perform the migration
    last_status = ""
    for status in tool.migrate(
            hf_token=hf_token,
            ms_token=ms_token,
            hf_repo_id=hf_repo_id,
            ms_repo_id=ms_repo_id,
            repo_type=repo_type,
            visibility=visibility,
            license_type=licence_type,
            chinese_name=chinese_name,
    ):
        last_status = status

    print("\n" + "=" * 50)
    print("Final Status:")
    print(last_status)


def main() -> None:
    """
    Main function demonstrating the migration tool usage.

    This function provides a concrete example of how to use the migrate function
    with specific repository parameters. It migrates the 'sentence-transformers/all-MiniLM-L6-v2'
    model from HuggingFace to ModelScope with public visibility and 'other' license type.

    :return: None
    :rtype: None
    """
    migrate(
        hf_repo_id='sentence-transformers/all-MiniLM-L6-v2',
        ms_repo_id='your-username/all-MiniLM-L6-v2',
        repo_type='model',
        visibility='public',
        licence_type='other'
    )


if __name__ == "__main__":
    main()
