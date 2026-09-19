# data

Tracked: small, frozen inputs. Not tracked: `raw/` (third-party downloads), `processed/` (rendered pools), `cache/`.
Third-party data is referenced by URL and hash and fetched by a script; none of it is redistributed here.

## Tracked files

| path | what | made by |
|---|---|---|
| `drift/sources_v0.json` | the 12 source formats of the simulated deployment at step 0; the legacy telemetry parser was written against these and frozen (`csrc/legacy/FREEZE`) | hand-written |
| `drift/smoke/*.yaml` | timeline configs of the legacy-only drift smoke (track, seed, payloads per source) | `scripts/w01_drift_smoke.py` |
| `curriculum/telemetry-stub-s0/` | task card, 200 validated format families, manifest (hashes, acceptance rate, pool sizes) of the stub-teacher curriculum | `scripts/w01_curriculum_smoke.py` |

## Pools (locked at creation)

| pool | n | families | seed | source |
|---|---|---|---|---|
| `tel-train` (smoke) | 50,000 | 180 | 0 | `curriculum/telemetry-stub-s0`, regenerated into `processed/` |
| `tel-dev` (smoke) | 5,000 | 20 | 0 | same run; families disjoint from `tel-train` |

## Third-party data (under `raw/`, never tracked)

| dataset | source | pinned | licence | fetch |
|---|---|---|---|---|
| LOGEVOL (Spark 2.4.0 / 3.0.3, Hadoop 2.10.2 / 3.3.3; 931,960 / 1,600,273 / 2,120,739 / 2,050,488 lines) | link in the README of `github.com/YintongHuo/EvLog` | `Logevol.zip`, 58,909,926 bytes, sha256 `1f82141b93ff120058a51f9305f669e77cb34223c9d792f15684193c00566285` | **none stated** (no licence file in the repository, none in the archive): not redistributed; labels we derive are released as scripts and as offsets keyed by session id and line index, without log text | `bash scripts/data/fetch_logevol.sh` |
| uap-core (regex list, fixtures, full history) | `github.com/ua-parser/uap-core` | commit `73e7340c3ed8055051607b296bf46ead7aa5f19e` (2026-05-15) | Apache-2.0 | `git clone https://github.com/ua-parser/uap-core.git data/raw/uap-core` |

Notes. The LOGEVOL archive is a gzipped tar despite its `.zip` name, and its files are Python pickles; they are read
through an allow-list unpickler (`static_student/datasets.py`). Each line is the message body only, without the log
header. Sessions carry anomaly labels; field labels are derived in this repository.
