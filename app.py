"""
HuggingFace to ModelScope Migration Tool

This Gradio app enables migration of models and datasets from HuggingFace to ModelScope.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional

import gradio as gr
from huggingface_hub import snapshot_download, HfApi
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


class MigrationTool:
    """Handles migration of models and datasets between HuggingFace and ModelScope."""

    def __init__(self):
        self.hf_api = None
        self.ms_api = None
        self.temp_dir = None

    def authenticate_hf(self, token: str) -> Tuple[bool, str]:
        """Authenticate with HuggingFace."""
        try:
            self.hf_api = HfApi(token=token)
            # Test the token
            self.hf_api.whoami(token=token)
            return True, "✓ HuggingFace authentication successful"
        except Exception as e:
            return False, f"✗ HuggingFace authentication failed: {str(e)}"

    def authenticate_ms(self, token: str) -> Tuple[bool, str]:
        """Authenticate with ModelScope."""
        try:
            self.ms_api = HubApi()
            self.ms_api.login(token)
            return True, "✓ ModelScope authentication successful"
        except Exception as e:
            return False, f"✗ ModelScope authentication failed: {str(e)}"

    def download_from_hf(
        self,
        repo_id: str,
        repo_type: str = "model"
    ) -> Tuple[bool, str, Optional[str]]:
        """Download a repository from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'username/repo-name')
            repo_type: Type of repository ('model' or 'dataset')

        Returns:
            Tuple of (success, message, local_path)
        """
        try:
            # Create temporary directory
            self.temp_dir = tempfile.mkdtemp(prefix="hf2ms_")

            # Download the repository
            local_path = snapshot_download(
                repo_id=repo_id,
                repo_type=repo_type,
                local_dir=self.temp_dir,
                local_dir_use_symlinks=False,
                token=self.hf_api.token if self.hf_api else None
            )

            return True, f"✓ Successfully downloaded {repo_type} from HuggingFace", local_path
        except Exception as e:
            return False, f"✗ Download failed: {str(e)}", None

    def upload_to_ms(
        self,
        local_path: str,
        repo_id: str,
        repo_type: str = "model",
        visibility: str = "public",
        license_type: str = "apache-2.0",
        chinese_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Upload a repository to ModelScope.

        Args:
            local_path: Local path to the repository
            repo_id: ModelScope repository ID (e.g., 'username/repo-name')
            repo_type: Type of repository ('model' or 'dataset')
            visibility: Repository visibility ('public' or 'private')
            license_type: License type
            chinese_name: Optional Chinese name for the repository

        Returns:
            Tuple of (success, message)
        """
        try:
            if not self.ms_api:
                return False, "✗ ModelScope not authenticated"

            # Determine visibility
            vis = ModelVisibility.PUBLIC if visibility == "public" else ModelVisibility.PRIVATE

            # Map license types
            license_map = {
                "apache-2.0": Licenses.APACHE_V2,
                "mit": Licenses.MIT,
                "gpl-3.0": Licenses.GPL_V3,
                "other": Licenses.OTHER,
            }
            lic = license_map.get(license_type.lower(), Licenses.APACHE_V2)

            # Create repository if it doesn't exist
            try:
                if repo_type == "model":
                    self.ms_api.create_model(
                        model_id=repo_id,
                        visibility=vis,
                        license=lic,
                        chinese_name=chinese_name
                    )
            except Exception as create_error:
                # Repository might already exist, continue to push
                pass

            # Push the model/dataset
            if repo_type == "model":
                self.ms_api.push_model(
                    model_id=repo_id,
                    model_dir=local_path
                )
            else:
                # For datasets, use similar approach
                self.ms_api.push_model(
                    model_id=repo_id,
                    model_dir=local_path
                )

            return True, f"✓ Successfully uploaded {repo_type} to ModelScope"
        except Exception as e:
            return False, f"✗ Upload failed: {str(e)}"

    def cleanup(self):
        """Clean up temporary files."""
        if self.temp_dir and os.path.exists(self.temp_dir):
            try:
                shutil.rmtree(self.temp_dir)
                self.temp_dir = None
            except Exception as e:
                print(f"Warning: Failed to clean up temporary directory: {e}")

    def migrate(
        self,
        hf_token: str,
        ms_token: str,
        hf_repo_id: str,
        ms_repo_id: str,
        repo_type: str,
        visibility: str,
        license_type: str,
        chinese_name: Optional[str] = None
    ) -> str:
        """Perform the complete migration process.

        Returns:
            Status message with migration progress
        """
        output = []

        # Validate inputs
        if not hf_token or not ms_token:
            return "✗ Error: Both HuggingFace and ModelScope tokens are required"

        if not hf_repo_id or not ms_repo_id:
            return "✗ Error: Both source and destination repository IDs are required"

        # Authenticate with HuggingFace
        output.append("🔐 Authenticating with HuggingFace...")
        success, msg = self.authenticate_hf(hf_token)
        output.append(msg)
        if not success:
            return "\n".join(output)

        # Authenticate with ModelScope
        output.append("\n🔐 Authenticating with ModelScope...")
        success, msg = self.authenticate_ms(ms_token)
        output.append(msg)
        if not success:
            return "\n".join(output)

        # Download from HuggingFace
        output.append(f"\n⬇️  Downloading {repo_type} from HuggingFace: {hf_repo_id}...")
        success, msg, local_path = self.download_from_hf(hf_repo_id, repo_type)
        output.append(msg)
        if not success:
            self.cleanup()
            return "\n".join(output)

        # Upload to ModelScope
        output.append(f"\n⬆️  Uploading {repo_type} to ModelScope: {ms_repo_id}...")
        success, msg = self.upload_to_ms(
            local_path,
            ms_repo_id,
            repo_type,
            visibility,
            license_type,
            chinese_name
        )
        output.append(msg)

        # Cleanup
        output.append("\n🧹 Cleaning up temporary files...")
        self.cleanup()
        output.append("✓ Cleanup complete")

        if success:
            output.append(f"\n✅ Migration completed successfully!")
            output.append(f"Your {repo_type} is now available at: https://www.modelscope.cn/models/{ms_repo_id}")
        else:
            output.append(f"\n❌ Migration failed")

        return "\n".join(output)


def create_interface():
    """Create the Gradio interface."""

    migration_tool = MigrationTool()

    with gr.Blocks(title="HuggingFace to ModelScope Migration Tool") as app:
        gr.Markdown("""
        # 🚀 HuggingFace to ModelScope Migration Tool

        Easily migrate your models and datasets from HuggingFace to ModelScope.

        ## 📋 Instructions:
        1. Get your **HuggingFace token** from: https://huggingface.co/settings/tokens
        2. Get your **ModelScope token** from: https://www.modelscope.cn/my/myaccesstoken
        3. Fill in the repository details below
        4. Click "Start Migration"
        """)

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🔑 Authentication")
                hf_token = gr.Textbox(
                    label="HuggingFace Token",
                    type="password",
                    placeholder="hf_...",
                    info="Your HuggingFace access token"
                )
                ms_token = gr.Textbox(
                    label="ModelScope Token",
                    type="password",
                    placeholder="Enter your ModelScope token",
                    info="Your ModelScope SDK token"
                )

            with gr.Column():
                gr.Markdown("### 📦 Repository Details")
                repo_type = gr.Radio(
                    choices=["model", "dataset"],
                    label="Repository Type",
                    value="model",
                    info="Select what you want to migrate"
                )
                visibility = gr.Radio(
                    choices=["public", "private"],
                    label="Visibility",
                    value="public",
                    info="Visibility of the repository on ModelScope"
                )

        with gr.Row():
            with gr.Column():
                hf_repo_id = gr.Textbox(
                    label="Source HuggingFace Repo ID",
                    placeholder="username/repo-name",
                    info="e.g., bert-base-uncased or username/my-model"
                )

            with gr.Column():
                ms_repo_id = gr.Textbox(
                    label="Destination ModelScope Repo ID",
                    placeholder="username/repo-name",
                    info="Your ModelScope username/repo-name"
                )

        with gr.Row():
            with gr.Column():
                license_type = gr.Dropdown(
                    choices=["apache-2.0", "mit", "gpl-3.0", "other"],
                    label="License",
                    value="apache-2.0",
                    info="License for the repository"
                )

            with gr.Column():
                chinese_name = gr.Textbox(
                    label="Chinese Name (Optional)",
                    placeholder="模型中文名称",
                    info="Optional Chinese name for the repository"
                )

        migrate_btn = gr.Button("🚀 Start Migration", variant="primary", size="lg")

        output = gr.Textbox(
            label="Migration Status",
            lines=15,
            max_lines=20,
            interactive=False,
            show_copy_button=True
        )

        migrate_btn.click(
            fn=migration_tool.migrate,
            inputs=[
                hf_token,
                ms_token,
                hf_repo_id,
                ms_repo_id,
                repo_type,
                visibility,
                license_type,
                chinese_name
            ],
            outputs=output
        )

        gr.Markdown("""
        ---
        ### 📝 Notes:
        - Large models may take some time to download and upload
        - Make sure you have enough disk space for temporary storage
        - Private repositories require appropriate token permissions
        - The tool will create the repository on ModelScope if it doesn't exist

        ### 🔗 Resources:
        - [HuggingFace Hub](https://huggingface.co/)
        - [ModelScope](https://www.modelscope.cn/)
        - [HuggingFace Token Settings](https://huggingface.co/settings/tokens)
        - [ModelScope Token Settings](https://www.modelscope.cn/my/myaccesstoken)
        """)

    return app


if __name__ == "__main__":
    app = create_interface()
    app.launch(share=False)
