from huggingface_hub import HfApi
from tqdm import tqdm

from example_usage import migrate

if __name__ == '__main__':
    hf_client = HfApi()

    repo_ids = []
    for repo_info in hf_client.list_datasets(author='deepghs'):
        if repo_info.id.endswith('_index') and 'character_index' not in repo_info.id \
                and 'danbooru2023_index' not in repo_info.id and 'e621-2024_index' not in repo_info.id \
                and 'e621-2024-webp-4Mpixel_index' not in repo_info.id \
                and 'deepghs/danbooru2023-webp-4Mpixel_index' not in repo_info.id \
                and 'deepghs/yandere2023_index' not in repo_info.id:
            repo_ids.append(repo_info.id)

    pg = tqdm(repo_ids)
    for repo_id in pg:
        pg.set_description(f'Syncing {repo_id!r}')
        migrate(
            hf_repo_id=repo_id,
            repo_type='dataset',
        )
