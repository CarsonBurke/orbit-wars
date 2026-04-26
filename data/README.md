# Data

Orbit Wars is a *simulation* competition — there's no static training dataset to download. The "data" we cache here is just the starter kit Kaggle ships:

```
data/raw/
  README.md      # the official "How to Play" doc
  agents.md      # the official getting-started guide
  main.py        # the nearest-planet sniper starter agent
```

Pull the latest:

```bash
python scripts/download.py
# or:
kaggle competitions download -c orbit-wars -p data/raw && unzip -o data/raw/orbit-wars.zip -d data/raw
```

Anything else under `data/` (replay JSONs, episode logs, generated trajectories) is local-only and `.gitignore`d.
