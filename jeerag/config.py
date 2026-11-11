"""Config for jeerag: env vars override config.json which overrides defaults."""
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = Path(os.environ.get("JEERAG_CONFIG", ROOT / "config.json"))
DEFAULTS = {
    "notes_dir": "/storage/emulated/0/JEE/Ambarish notes",
    "data_dir": str(ROOT / "data"),
    "top_k": 8,
    "port": 8613,
    "llm_base_url": "",
    "llm_model": "",
    "llm_api_key": "",
    "llm_keys": [],
    "llm_vision_model": "",
    "spaced_repeat": True,
}


def load():
    cfg = dict(DEFAULTS)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text()))
        except Exception:
            pass
    for key in cfg:
        env = os.environ.get("JEERAG_" + key.upper())
        if env:
            cfg[key] = int(env) if key in ("top_k", "port") else env
    return cfg


def save(updates):
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            cfg = json.loads(CONFIG_PATH.read_text())
        except Exception:
            cfg = {}
    cfg.update(updates)
    CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    return cfg
