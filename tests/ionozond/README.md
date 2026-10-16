# ionozond test data

Byte-for-byte copies from the ionozond repository (nikita-konkin/ionozond,
commit 7612f02). Do not edit them: the tests that read them prove that the
agent reads, edits and regenerates exactly what the Qt console does.

| File here | ionozond source | What it is |
|---|---|---|
| `golden_config.ini`, `golden_schedule.ini` | `tests/golden/` | the settings the original dsChirp binary left behind, written by `QSettings` |
| `golden_chirp_config.py.txt` | `tests/golden/chirp_config.py` | what that binary generated from them; `.txt` because it is not valid Python (`tb = `) |
| `fixture_config.ini`, `fixture_schedule.ini` | `tests/fixtures/` | the console's test fixtures |
| `console_config.ini`, `console_schedule.ini` | `res/ini/` | the defaults the console installs on first start, with quoted values |

Still to check on the station: a round trip of the real
`~/.config/dsChirp/schedule.ini` and `config.ini` from the Yoshkar-Ola laptop.
