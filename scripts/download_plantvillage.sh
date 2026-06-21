#!/usr/bin/env bash
set -euo pipefail

DESTINATION="data/raw/PlantVillage"
if [[ -d "${DESTINATION}" ]] && find "${DESTINATION}" -type f -print -quit | grep -q .; then
  echo "Dataset destination is not empty: ${DESTINATION}"
  echo "Move or remove it explicitly before downloading again."
  exit 1
fi

mkdir -p data/raw
rm -rf "${DESTINATION}"
git clone --depth 1 --filter=blob:none --no-checkout \
  https://github.com/spMohanty/PlantVillage-Dataset.git "${DESTINATION}"
git -C "${DESTINATION}" sparse-checkout init --no-cone
cat > "${DESTINATION}/.git/info/sparse-checkout" <<'EOF'
/raw/color/Tomato___healthy/
/raw/color/Tomato___Early_blight/
/raw/color/Tomato___Late_blight/
/raw/color/Tomato___Leaf_Mold/
EOF
git -C "${DESTINATION}" checkout
echo "PlantVillage downloaded to ${DESTINATION}"
