from huggingface_hub import HfApi
from tqdm import tqdm

from example_usage import migrate

if __name__ == '__main__':
    hf_client = HfApi()

    repo_ids = []
    for repo_info in hf_client.list_models(author='deepghs'):
        if repo_info.private:
            repo_ids.append(repo_info.id)

    pg = tqdm(repo_ids)
    for repo_id in pg:
        pg.set_description(f'Syncing {repo_id!r}')
        migrate(
            hf_repo_id=repo_id,
            repo_type='model',
        )
