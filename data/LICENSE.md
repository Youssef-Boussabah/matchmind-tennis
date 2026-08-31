# Data License

**This notice applies to the third-party data files in this `data/` directory. It does not
cover the project's own source code or documentation; those are covered separately by the
[MIT License](../LICENSE) at the repository root.**

## Files covered

- `data/all/atp_matches_1991.csv` … `atp_matches_2026.csv` — 36 yearly ATP match files
- `data/atp_players.csv` — the ATP player reference table
- `data/matches_data_dictionary.txt` — the data dictionary distributed with the dataset

## Attribution and terms

These files come from Jeff Sackmann's `tennis_atp` dataset. The dataset's own README states
that the tennis databases and files are made available under the **Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 International** licence (CC BY-NC-SA 4.0).

- Attribution: **Jeff Sackmann / Tennis Abstract**
- Licence: **CC BY-NC-SA 4.0** — <https://creativecommons.org/licenses/by-nc-sa/4.0/>
- Original source repository: <https://github.com/JeffSackmann/tennis_atp>

The original repository was unavailable at the time this release was prepared; it returned
404. The 1991–2024 match files are the preserved copies originally used by this project. The
2025 and partial-2026 files were added from the pinned redistribution snapshot recorded in
[SOURCE_SNAPSHOT.txt](SOURCE_SNAPSHOT.txt); the older files were verified content-identical to
that snapshot apart from line endings. Every match file here is the same Jeff Sackmann /
Tennis Abstract data under the licence above — no second data provider is blended in — and
nothing in the project downloads or re-fetches them at build or runtime.

Anyone redistributing or building on this data should read the licence text at the URL above
and apply its terms themselves.
