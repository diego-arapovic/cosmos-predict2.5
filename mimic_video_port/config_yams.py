"""Custom `--config` entry for `scripts.train` that registers the YAMS world2action experiments.

  torchrun --nproc_per_node=N -m scripts.train --config=mimic_video_port/config_yams.py -- experiment=yams_smoke

`scripts.train` does `importlib.import_module(<this>).make_config()`. Importing this module first runs
`yams_experiment`'s `cs.store(...)` (registering `yams_smoke`/`yams`), then `make_config` (re-exported from
the stock video2world config) composes the base defaults + our experiment. No stock 2.5 files are edited.
"""
# Re-export the stock make_config (registers all base defaults + groups when called).
from cosmos_predict2._src.predict2.configs.video2world.config import make_config  # noqa: F401

# Importing this registers the yams_smoke / yams experiments via cs.store (must happen before compose).
import mimic_video_port.world2action.configs.yams_experiment  # noqa: F401,E402
