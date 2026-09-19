#!/usr/bin/env bash
# Download LOGEVOL (Spark 2.4.0 / 3.0.3, Hadoop 2.10.2 / 3.3.3) from the link in the dataset authors' README, verify
# the archive hash, and unpack it under data/raw/logevol/. The dataset carries no licence statement, so nothing from
# it is redistributed here: this script and the hash are the reference.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/../.." && pwd)"; DEST="$REPO/data/raw/logevol"; mkdir -p "$DEST"
URL="https://smu-my.sharepoint.com/:u:/g/personal/ythuo_smu_edu_sg/IQBsHUy0DktJQYzm5cozeQvnAcwWZjljocP5HJ5mTer4SfI"
SHA="1f82141b93ff120058a51f9305f669e77cb34223c9d792f15684193c00566285"
JAR="$(mktemp)"; trap 'rm -f "$JAR"' EXIT
if [ ! -f "$DEST/Logevol.zip" ]; then
  curl -sL -A "Mozilla/5.0" -c "$JAR" -b "$JAR" -o /dev/null "$URL"          # the share link sets a session cookie first
  curl -sL -A "Mozilla/5.0" -c "$JAR" -b "$JAR" -o "$DEST/Logevol.zip" "$URL?download=1"
fi
echo "$SHA  $DEST/Logevol.zip" | shasum -a 256 -c -
tar -xzf "$DEST/Logevol.zip" -C "$DEST"                                      # a gzipped tar despite the extension
find "$DEST" -name ".DS_Store" -delete
ls "$DEST/Logevol"
