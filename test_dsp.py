import os

from huggingface_hub import HfApi
from modelscope import HubApi
from tqdm import tqdm

from example_usage import migrate

if __name__ == '__main__':
    hf_client = HfApi()
    ms_client = HubApi()
    if os.environ.get('MODELSCOPE_TOKEN'):
        ms_client.login(access_token=os.environ['MODELSCOPE_TOKEN'])

    repo_ids = []
    for repo_info in tqdm(hf_client.list_datasets(author='deepghs'), desc='Scanning'):
        if repo_info.private and not ms_client.repo_exists(repo_id=repo_info.id, repo_type='dataset'):
            repo_ids.append(repo_info.id)

    pg = tqdm(repo_ids)
    for repo_id in pg:
        pg.set_description(f'Syncing {repo_id!r}')
        migrate(
            hf_repo_id=repo_id,
            repo_type='dataset',
        )
