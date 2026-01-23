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

    repo_ids = [
        'deepghs/few_shots'
    ]


    pg = tqdm(repo_ids)
    for repo_id in pg:
        pg.set_description(f'Syncing {repo_id!r}')
        migrate(
            hf_repo_id=repo_id,
            repo_type='dataset',
        )
