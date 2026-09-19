"""Resume the exact saved partial batch under an amended execution allowance."""
import argparse
from pathlib import Path
from .core import sha, IntegrityError


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--home', required=True); p.add_argument('--folder', required=True)
    p.add_argument('--state-sha', required=True); p.add_argument('--training-recipe-sha', required=True); a = p.parse_args()
    folder = Path(a.folder)
    if sha(folder/'state.pt') != a.state_sha or sha(folder/'recipe.json') != a.training_recipe_sha:
        raise IntegrityError('Saved partial batch or frozen training recipe changed before resume')
    from .train_robust_block_recompute import train
    train(a.home, folder)
