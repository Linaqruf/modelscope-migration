"""
HuggingFace to ModelScope Migration Tool

This Gradio app enables migration of models and datasets from HuggingFace to ModelScope.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Tuple, Optional
import argparse

import gradio as gr
from huggingface_hub import snapshot_download, HfApi
from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility, DatasetVisibility
import sys
import io
import threading
import queue
import time
import re

# Regex to match ANSI escape codes (like [A, [2K, etc.)
ANSI_ESCAPE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')


class MigrationTool:
    """Handles migration of models and datasets between HuggingFace and ModelScope."""

    def __init__(self):
        self.temp_dir = None

    def download_from_hf(
        self,
        repo_id: str,
        repo_type: str = "model",
        token: Optional[str] = None
    ) -> Tuple[bool, str, Optional[str]]:
        """Download a repository from HuggingFace.

        Args:
            repo_id: HuggingFace repository ID (e.g., 'username/repo-name')
            repo_type: Type of repository ('model' or 'dataset')
            token: HuggingFace authentication token

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
                token=token
            )

            return True, f"✓ Successfully downloaded {repo_type} from HuggingFace", local_path
        except Exception as e:
            return False, f"✗ Download failed: {str(e)}", None

    def upload_to_ms(
        self,
        local_path: str,
        repo_id: str,
        token: str,
        repo_type: str = "model",
        visibility: str = "public",
        license_type: str = "apache-2.0",
        chinese_name: Optional[str] = None
    ) -> Tuple[bool, str]:
        """Upload a repository to ModelScope.

        Args:
            local_path: Local path to the repository
            repo_id: ModelScope repository ID (e.g., 'username/repo-name')
            token: ModelScope authentication token
            repo_type: Type of repository ('model' or 'dataset')
            visibility: Repository visibility ('public' or 'private')
            license_type: License type
            chinese_name: Optional Chinese name for the repository

        Returns:
            Tuple of (success, message)
        """
        try:
            # Clean and validate token
            token = token.strip()
            if not token:
                return False, "✗ ModelScope token is empty"

            # Create HubApi instance and login explicitly
            api = HubApi()
            try:
                api.login(token)
            except Exception as login_error:
                return False, f"✗ ModelScope Login failed: {str(login_error)}\n\n💡 Tip: Ensure you are using an 'SDK Token' from https://www.modelscope.cn (NOT modelscope.ai). The token usually starts with 'ms-'."

            # Map license types
            license_map = {
                "apache-2.0": Licenses.APACHE_V2,
                "mit": Licenses.MIT,
                "gpl-2.0": Licenses.GPL_V2,
                "gpl-3.0": Licenses.GPL_V3,
                "lgpl-2.1": Licenses.LGPL_V2_1,
                "lgpl-3.0": Licenses.LGPL_V3,
                "afl-3.0": Licenses.AFL_V3,
                "ecl-2.0": Licenses.ECL_V2,
                "other": None,
            }
            lic = license_map.get(license_type.lower(), Licenses.APACHE_V2)

            # Check if repository exists
            repo_exists = api.repo_exists(repo_id=repo_id, repo_type=repo_type, token=token)
            
            # Create repository if it doesn't exist
            # Important: We must create with the correct visibility BEFORE upload_folder,
            # because upload_folder will create a public repo by default if repo doesn't exist
            if not repo_exists:
                try:
                    if repo_type == "model":
                        # Determine visibility for models (1=private, 5=public)
                        vis = ModelVisibility.PUBLIC if visibility == "public" else ModelVisibility.PRIVATE
                        # Build parameters, only include license if not None
                        create_params = {
                            "model_id": repo_id,
                            "visibility": vis,
                            "token": token,
                        }
                        if lic is not None:
                            create_params["license"] = lic
                        if chinese_name:
                            create_params["chinese_name"] = chinese_name
                        api.create_model(**create_params)
                    else:
                        # Determine visibility for datasets (1=private, 5=public)
                        vis = DatasetVisibility.PUBLIC if visibility == "public" else DatasetVisibility.PRIVATE
                        # For datasets, need to split repo_id into namespace and name
                        parts = repo_id.split('/')
                        if len(parts) != 2:
                            return False, f"✗ Invalid dataset ID format: {repo_id}. Must be 'namespace/name'"
                        namespace, dataset_name = parts
                        # Build parameters, only include license if not None
                        create_params = {
                            "dataset_name": dataset_name,
                            "namespace": namespace,
                            "visibility": vis,
                        }
                        if lic is not None:
                            create_params["license"] = lic
                        if chinese_name:
                            create_params["chinese_name"] = chinese_name
                        api.create_dataset(**create_params)
                except Exception as create_error:
                    error_msg = str(create_error)
                    # Only ignore if repo already exists (race condition)
                    if "already exists" not in error_msg.lower():
                        return False, f"✗ Failed to create repository: {error_msg}"

            # Push the model/dataset
            if repo_type == "model":
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=local_path,
                    token=token,
                )
            else:
                # For datasets, use upload_folder with repo_type='dataset'
                api.upload_folder(
                    repo_id=repo_id,
                    folder_path=local_path,
                    token=token,
                    repo_type="dataset"
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
        chinese_name: Optional[str] = None,
        progress=None
    ) -> str:
        """Perform the complete migration process with real-time console log capture."""
        
        # If no progress tracker is provided (CLI mode), we just skip progress updates
        # Gradio will pass its own tracker when called from the UI
        def update_progress(val, desc=""):
            if progress:
                progress(val, desc=desc)

        log_queue = queue.Queue()
        output_lines = []
        
        # Helper to capture console output and send it to the queue
        class StreamToQueue(io.StringIO):
            def __init__(self, original_stream, q):
                super().__init__()
                self.original_stream = original_stream
                self.q = q
            def write(self, s):
                if s:
                    # Write to original stream (console) and our queue (Gradio)
                    self.original_stream.write(s)
                    self.original_stream.flush()
                    self.q.put(s)
            def flush(self):
                self.original_stream.flush()

        def update_output():
            """Gather all pending messages from the queue and return the full status."""
            new_data = False
            while not log_queue.empty():
                try:
                    raw_msg = log_queue.get_nowait()
                    # 1. Strip ANSI escape codes (those [A, [m, etc.)
                    msg = ANSI_ESCAPE.sub('', raw_msg)
                    if not msg:
                        continue

                    # 2. Process the message line by line
                    # We handle \r by treating it as a signal to potentially overwrite the last line
                    # We handle \n as a signal to start a new line
                    for line in msg.replace('\r', '\n').split('\n'):
                        clean_line = line.strip()
                        if not clean_line:
                            continue

                        # 3. Smart Progress Bar Handling
                        # Identify if this line is a progress bar update
                        # Progress bars usually look like: "Label: 45%|### | ..."
                        is_progress = '%' in clean_line and '|' in clean_line and ('[' in clean_line or ']' in clean_line)
                        
                        if is_progress:
                            # Extract the label (everything before the progress bar/percentage)
                            # This helps us identify WHICH progress bar to update
                            label = clean_line.split('|')[0].split('%')[0].strip()
                            # If the label ends with a number (like '45'), try to get the text before it
                            label = re.sub(r'\d+$', '', label).strip()
                            
                            found = False
                            # Look at the last few lines to see if we're updating an existing bar
                            # We only look back ~10 lines to keep it fast
                            for i in range(len(output_lines) - 1, max(-1, len(output_lines) - 11), -1):
                                if label and label in output_lines[i] and ('%' in output_lines[i] or '|' in output_lines[i]):
                                    output_lines[i] = clean_line
                                    found = True
                                    new_data = True
                                    break
                            
                            if not found:
                                output_lines.append(clean_line)
                                new_data = True
                        else:
                            # Regular log message
                            output_lines.append(clean_line)
                            new_data = True

                except queue.Empty:
                    break
            
            # Keep the output box from growing infinitely (last 1000 lines)
            if len(output_lines) > 1000:
                output_lines[:] = output_lines[-1000:]
                
            return "\n".join(output_lines), new_data

        # Thread-safe migration execution storage
        results = {"success": False, "message": "", "finished": False}

        def run_migration():
            try:
                # Clean inputs
                update_progress(0, desc="Validating inputs...")
                nonlocal hf_token, ms_token, hf_repo_id, ms_repo_id
                hf_token = hf_token.strip() if hf_token else ""
                ms_token = ms_token.strip() if ms_token else ""
                hf_repo_id = hf_repo_id.strip() if hf_repo_id else ""
                ms_repo_id = ms_repo_id.strip() if ms_repo_id else ""

                if not hf_token or not ms_token or not hf_repo_id or not ms_repo_id:
                    results["message"] = "✗ Error: All tokens and repository IDs are required"
                    results["finished"] = True
                    return

                if "/" not in ms_repo_id:
                    results["message"] = "✗ Error: ModelScope Repo ID must include your namespace (e.g., 'username/repo-name')"
                    results["finished"] = True
                    return

                # 1. Download
                update_progress(0.1, desc="Downloading from HuggingFace...")
                print(f"⬇️  Starting download from HuggingFace: {hf_repo_id}...")
                success, msg, local_path = self.download_from_hf(hf_repo_id, repo_type, hf_token)
                print(msg)
                if not success:
                    results["message"] = msg
                    results["finished"] = True
                    return

                # 2. Upload
                update_progress(0.4, desc="Uploading to ModelScope...")
                print(f"\n⬆️  Starting upload to ModelScope: {ms_repo_id}...")
                success, msg = self.upload_to_ms(
                    local_path, 
                    ms_repo_id, 
                    ms_token, 
                    repo_type, 
                    visibility, 
                    license_type, 
                    chinese_name
                )
                print(msg)
                results["success"] = success
                results["message"] = msg

            except Exception as e:
                results["message"] = f"✗ Unexpected error: {str(e)}"
            finally:
                print("\n🧹 Cleaning up temporary files...")
                self.cleanup()
                print("✓ Cleanup complete")
                results["finished"] = True

        # Redirect stdout and stderr to our queue
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = StreamToQueue(sys.stdout, log_queue)
        sys.stderr = StreamToQueue(sys.stderr, log_queue)

        try:
            # Start the migration in a background thread
            thread = threading.Thread(target=run_migration)
            thread.start()

            # Continuously yield updates until the migration thread completes
            while not results["finished"]:
                current_status, updated = update_output()
                if updated:
                    yield current_status
                time.sleep(0.1)

            # Final capture of any remaining logs
            final_status, _ = update_output()
            
            # Append final results
            if results["success"]:
                update_progress(1.0, desc="Completed")
                final_status += f"\n\n✅ Migration completed successfully!"
                final_status += f"\nYour {repo_type} is available at: https://www.modelscope.cn/models/{ms_repo_id}"
            else:
                update_progress(1.0, desc="Failed")
                final_status += f"\n\n❌ Migration failed: {results['message']}"
            
            yield final_status

        finally:
            # CRITICAL: Restore original streams so we don't break the whole app
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def create_interface():
    """Create the Gradio interface."""

    migration_tool = MigrationTool()

    with gr.Blocks(title="HuggingFace to ModelScope Migration Tool") as app:
        gr.Markdown("""
        # 🚀 HuggingFace to ModelScope Migration Tool

        Easily migrate your models and datasets from HuggingFace to ModelScope.

        ## 📋 Instructions:
        1. Get your **HuggingFace token** from: https://huggingface.co/settings/tokens
        2. Get your **ModelScope SDK token** from: https://www.modelscope.cn/my/myaccesstoken
           * **Note**: Use tokens from **modelscope.cn** (Chinese site). The international site (modelscope.ai) is not currently supported by the SDK.
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
                    placeholder="Enter your ModelScope SDK token",
                    info="Use your SDK token from modelscope.cn (usually starts with 'ms-')"
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
                    choices=[
                        "apache-2.0", 
                        "mit", 
                        "gpl-2.0", 
                        "gpl-3.0", 
                        "lgpl-2.1", 
                        "lgpl-3.0", 
                        "afl-3.0", 
                        "ecl-2.0",
                        "other"
                    ],
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
            interactive=False
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
    parser = argparse.ArgumentParser(description="HuggingFace to ModelScope Migration Tool")
    
    # Mode selection
    parser.add_argument("--cli", action="store_true", help="Run in CLI mode instead of Gradio UI")
    
    # Gradio options
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Host for Gradio app")
    parser.add_argument("--port", type=int, default=7860, help="Port for Gradio app")
    parser.add_argument("--share", action="store_true", help="Create a public link for Gradio app")
    
    # CLI options (only used if --cli is set)
    parser.add_argument("--hf-token", type=str, help="HuggingFace access token")
    parser.add_argument("--ms-token", type=str, help="ModelScope access token")
    parser.add_argument("--hf-repo", type=str, help="Source HuggingFace repo ID")
    parser.add_argument("--ms-repo", type=str, help="Destination ModelScope repo ID")
    parser.add_argument("--type", type=str, choices=["model", "dataset"], default="model", help="Repository type")
    parser.add_argument("--visibility", type=str, choices=["public", "private"], default="public", help="Repo visibility")
    parser.add_argument("--license", type=str, default="apache-2.0", help="Repo license")
    parser.add_argument("--chinese-name", type=str, help="Optional Chinese name for the repo")
    
    args = parser.parse_args()
    
    if args.cli:
        if not all([args.hf_token, args.ms_token, args.hf_repo, args.ms_repo]):
            print("✗ Error: CLI mode requires --hf-token, --ms-token, --hf-repo, and --ms-repo")
            sys.exit(1)
            
        tool = MigrationTool()
        print(f"🚀 Starting CLI Migration: {args.hf_repo} -> {args.ms_repo}")
        
        # In CLI mode, we just consume the generator
        last_status = ""
        for status in tool.migrate(
            args.hf_token,
            args.ms_token,
            args.hf_repo,
            args.ms_repo,
            args.type,
            args.visibility,
            args.license,
            args.chinese_name
        ):
            last_status = status
            
        print("\n" + "="*50)
        print("Final Status:")
        print(last_status)
    else:
        app = create_interface()
        app.launch(
            server_name=args.host,
            server_port=args.port,
            share=args.share
        )
