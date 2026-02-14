import pandas as pd
from huggingface_hub import HfApi
from tqdm import tqdm

from example_usage import migrate

if __name__ == '__main__':
    hf_client = HfApi()

    df = pd.read_csv('deepghs_t1+.csv')
    df = df[df['repo_type'].isin({'model', 'dataset'})]

    pg = tqdm(list(zip(df['repo_id'], df['repo_type'])))
    for repo_id, repo_type in pg:
        if not hf_client.repo_exists(repo_id=repo_id, repo_type=repo_type):
            continue
        if repo_id == 'deepghs/character_index':
            continue
        pg.set_description(f'Syncing {repo_id!r} ({repo_type})')
        migrate(
            hf_repo_id=repo_id,
            repo_type=repo_type,
        )
