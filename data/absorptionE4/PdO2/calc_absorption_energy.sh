#!/bin/bash

CO_DIR="./CO"
SURFACE_DIR="./surface"
ADSORBED_DIR="./absorbed"

get_e0() {
    val=$(grep "E0=" "$1/OSZICAR" | tail -n 1 | sed -n 's/.*E0= *\([-0-9.E+]*\).*/\1/p')
    printf "%f\n" "$val"
}

E1=$(get_e0 "$CO_DIR")
E2=$(get_e0 "$SURFACE_DIR")
E3=$(get_e0 "$ADSORBED_DIR")

echo "E1 (CO): $E1"
echo "E2 (Surface): $E2"
echo "E3 (Adsorbed): $E3"

if [[ -z "$E1" || -z "$E2" || -z "$E3" ]]; then
    echo "❌ 提取能量失败，请检查 OSZICAR 文件格式或路径是否正确。"
    exit 1
fi

absorption_energy=$(echo "$E3 - $E1 - $E2" | bc -l)
echo "✅ 吸附能: $absorption_energy eV"
