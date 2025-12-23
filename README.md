# Tipster: EuroLeague walk-forward starter

Ovo je mali starter za walk-forward modeliranje EuroLeague utakmica:

- Povlačenje podataka sa zvaničnog `euroleague-api` wrappera (game metadata, boxscore/pbp hookovi).
- Povlačenje kvota sa The Odds API (moneyline/spread/totals) za sanity-check modela.
- Izgradnja rolling feature-a po timovima i konstruktovanje home vs away feature diffa.
- Walk-forward (kolo-po-kolo) treniranje logistic regression modela za pobednika.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Struktura

```
src/elwf
  data/          # fetch_euroleague.py, fetch_odds.py
  features/      # build_game_features.py
  models/        # train_game.py (winner model + walk-forward)
  backtest/      # walkforward orchestrator
  cli.py         # komanda elwf
```

## Korišćenje (primer workflow)

1) Skini EuroLeague utakmice i boxscore (primer stub)
```python
from pathlib import Path
import pandas as pd
from elwf.data.fetch_euroleague import fetch_games

# primer: sezona 2023
meta = fetch_games(2023)
meta.to_csv("data/raw/games_2023.csv", index=False)
```

2) Spremi feature-e po timu i spoji u home/away diff

```python
import pandas as pd
from elwf.features.build_game_features import build_game_features

# Pretpostavimo da ste boxscore transformisali u jedan red po timu/utakmici
team_games = pd.read_parquet("data/processed/team_games.parquet")
features = build_game_features(team_games, window=5)
features.to_csv("data/processed/game_features.csv", index=False)
```

`team_games` treba da sadrži kolone:
- `season`, `round`, `game_code`
- `team_code`, `opponent_code`, `is_home`, `tipoff` (datetime)
- metrike iz `ROLLING_METRICS` (vidi modul)

3) Walk-forward treniranje i evaluacija

```bash
elwf data/processed/games.csv data/processed/game_features.csv --out artifacts/walk.csv
```

`data/processed/games.csv` treba da ima makar kolone: `season,round,game_code,y_home_win`.

Rezultati:
- `artifacts/walk.csv` sa kolonama `season,round,game_code,y_home_win,p_home_win`
- `artifacts/walk.metrics.json` sa metrikama (`logloss`, `brier`, `acc@0.5`).

### Brzi start sa sample CSV fajlovima

U repo-u postoje primeri u folderu `examples/`:

- `examples/sample_games.csv`
- `examples/sample_game_features.csv`
- `examples/sample_upcoming.csv` (opciono za buduća kola)

Pokretanje bez upcoming:

```bash
elwf examples/sample_games.csv examples/sample_game_features.csv --out artifacts/walk.csv
```

Pokretanje sa upcoming (predikcije za sledeće kolo):

```bash
elwf examples/sample_games.csv examples/sample_game_features.csv \
     --upcoming examples/sample_upcoming.csv \
     --out artifacts/walk.csv
```

Output:
- `artifacts/walk.csv` – istorijske predikcije
- `artifacts/walk.metrics.json` – metrike na istoriji
- `artifacts/walk.per_round.csv` – metrike po kolu
- `artifacts/walk.upcoming.csv` – predikcije za buduće kolo (ako `--upcoming`)

Napomena: sample set je mali, pa je potrebno spustiti prag za veličinu treninga:
```bash
elwf examples/sample_games.csv examples/sample_game_features.csv \
     --upcoming examples/sample_upcoming.csv \
     --out artifacts/walk.csv \
     --min-train 1 \
     --debug
```
Za realne podatke zadržite podrazumevani `--min-train 20` ili više.

## The Odds API (kvote)

Minimalan primer dohvaćanja kvota:
```python
import pandas as pd
from elwf.data.fetch_odds import fetch_odds
from elwf.settings import OddsApiSettings

settings = OddsApiSettings(api_key="YOUR_API_KEY")
odds_df = fetch_odds(settings)
odds_df.to_csv("data/raw/odds.csv", index=False)
```

`price` je decimalna kvota; implied probability: `1/price` (pomoćna funkcija u modulu).

## Napomene
- Ovo je kostur: treba dodati ETL za boxscore → timski agregati, i eventualno dodatne modele (spread, totals, player props) po želji.
- Walk-forward petlja izbegava leak jer trenira samo na prethodnim kolima.
- Preporuka: validirati kalibraciju (Brier/logloss), voditi evidenciju o “value” prema tržištu.
