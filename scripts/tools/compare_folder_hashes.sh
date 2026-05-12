#!/usr/bin/env bash

set -Eeuo pipefail

DEFAULT_SOURCE_SHARE_ROOT="/run/user/1000/gvfs/smb-share:server=haji-pr4764gw-4.local,share=raid0"
DEFAULT_BACKUP_SUBDIR="BACKUP_TMP"

INPUT_SOURCE="${1:-$DEFAULT_SOURCE_SHARE_ROOT}"
TARGET_ROOT="${2:-/media/haji/HDD-2T}"
OUTPUT_DIR="${3:-}"
BACKUP_SUBDIR="${4:-$DEFAULT_BACKUP_SUBDIR}"
REUSE_MANIFESTS="${REUSE_MANIFESTS:-1}"

if [[ "$(basename -- "$INPUT_SOURCE")" == "$BACKUP_SUBDIR" ]]; then
  SOURCE_CONTENT_ROOT="$INPUT_SOURCE"
  SOURCE_SHARE_ROOT="$(dirname -- "$INPUT_SOURCE")"
else
  SOURCE_SHARE_ROOT="$INPUT_SOURCE"
  SOURCE_CONTENT_ROOT="$SOURCE_SHARE_ROOT/$BACKUP_SUBDIR"
fi

timestamp() {
  date '+%Y-%m-%d %H:%M:%S'
}

usage() {
  cat <<'EOF'
用法:
  bash scripts/tools/compare_folder_hashes.sh [SOURCE_SHARE_ROOT|BACKUP_TMP_PATH] [TARGET_ROOT] [OUTPUT_DIR] [BACKUP_SUBDIR]

说明:
  1. 默认源目录:
     /run/user/1000/gvfs/smb-share:server=haji-pr4764gw-4.local,share=raid0
     默认会取其中的 BACKUP_TMP 与 /media/haji/HDD-2T 做同名项匹配。
  2. 比较规则:
     - 比较 BACKUP_TMP 和目标目录下“同名文件/同名文件夹”
     - 额外比较 raid0 根目录与目标目录下“同名文件”
     - 不再处理仅一侧存在的项目
  3. 同名文件直接算 sha256；同名文件夹递归生成 manifest 后再比较 manifest 的 sha256。
  4. 如果 OUTPUT_DIR 已存在，默认复用已有 manifest，可通过:
     REUSE_MANIFESTS=0 bash ...
     强制重算。
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

for cmd in find sort sha256sum diff stat awk readlink; do
  if ! command -v "$cmd" >/dev/null 2>&1; then
    echo "缺少命令: $cmd" >&2
    exit 1
  fi
done

if [[ ! -d "$SOURCE_SHARE_ROOT" ]]; then
  echo "源共享目录不存在: $SOURCE_SHARE_ROOT" >&2
  exit 1
fi

if [[ ! -d "$SOURCE_CONTENT_ROOT" ]]; then
  echo "源内容目录不存在: $SOURCE_CONTENT_ROOT" >&2
  exit 1
fi

if [[ ! -d "$TARGET_ROOT" ]]; then
  echo "目标目录不存在: $TARGET_ROOT" >&2
  exit 1
fi

if [[ -z "$OUTPUT_DIR" ]]; then
  RUN_ID="$(date '+%Y%m%d_%H%M%S')"
  OUTPUT_DIR="$(pwd)/hash_compare_results/$RUN_ID"
fi

MANIFEST_SRC_DIR="$OUTPUT_DIR/manifests/source"
MANIFEST_TGT_DIR="$OUTPUT_DIR/manifests/target"
DIFF_DIR="$OUTPUT_DIR/diffs"
SUMMARY_TSV="$OUTPUT_DIR/summary.tsv"
REPORT_TXT="$OUTPUT_DIR/report.txt"
META_TXT="$OUTPUT_DIR/metadata.txt"
RUN_LOG="$OUTPUT_DIR/run.log"

mkdir -p "$MANIFEST_SRC_DIR" "$MANIFEST_TGT_DIR" "$DIFF_DIR"

exec > >(tee -a "$RUN_LOG") 2>&1

echo "[$(timestamp)] 开始比较"
echo "SOURCE_SHARE_ROOT=$SOURCE_SHARE_ROOT"
echo "SOURCE_CONTENT_ROOT=$SOURCE_CONTENT_ROOT"
echo "TARGET_ROOT=$TARGET_ROOT"
echo "OUTPUT_DIR=$OUTPUT_DIR"
echo "REUSE_MANIFESTS=$REUSE_MANIFESTS"
echo

cat > "$META_TXT" <<EOF
started_at=$(timestamp)
source_share_root=$SOURCE_SHARE_ROOT
source_content_root=$SOURCE_CONTENT_ROOT
target_root=$TARGET_ROOT
output_dir=$OUTPUT_DIR
reuse_manifests=$REUSE_MANIFESTS
EOF

printf 'scope\titem_type\tstatus\tsource_hash\ttarget_hash\tsource_path\ttarget_path\tdetails\n' > "$SUMMARY_TSV"
{
  echo "Hash Compare Report"
  echo "started_at: $(timestamp)"
  echo "source_share_root: $SOURCE_SHARE_ROOT"
  echo "source_content_root: $SOURCE_CONTENT_ROOT"
  echo "target_root: $TARGET_ROOT"
  echo "output_dir: $OUTPUT_DIR"
  echo
} > "$REPORT_TXT"

build_manifest() {
  local dir_path="$1"
  local manifest_path="$2"
  local tmp_path
  tmp_path="$(mktemp)"

  (
    cd "$dir_path"
    find . -mindepth 1 -print0 | sort -z |
      while IFS= read -r -d '' rel; do
        rel="${rel#./}"
        if [[ -L "$rel" ]]; then
          printf 'L\t%s\t%s\n' "$rel" "$(readlink -- "$rel")"
        elif [[ -d "$rel" ]]; then
          printf 'D\t%s\n' "$rel"
        elif [[ -f "$rel" ]]; then
          printf 'F\t%s\t%s\t%s\n' \
            "$rel" \
            "$(stat -c '%s' -- "$rel")" \
            "$(sha256sum -- "$rel" | awk '{print $1}')"
        else
          printf 'O\t%s\n' "$rel"
        fi
      done
  ) > "$tmp_path"

  mv "$tmp_path" "$manifest_path"
}

manifest_hash() {
  sha256sum -- "$1" | awk '{print $1}'
}

write_summary() {
  local scope="$1"
  local item_type="$2"
  local status="$3"
  local source_hash="$4"
  local target_hash="$5"
  local source_path="$6"
  local target_path="$7"
  local details="$8"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$scope" "$item_type" "$status" "$source_hash" "$target_hash" "$source_path" "$target_path" "$details" \
    >> "$SUMMARY_TSV"
}

item_type_of() {
  local path="$1"
  if [[ -L "$path" ]]; then
    echo "symlink"
  elif [[ -d "$path" ]]; then
    echo "dir"
  elif [[ -f "$path" ]]; then
    echo "file"
  else
    echo "other"
  fi
}

compare_dir_scope() {
  local scope="$1"
  local source_path="$2"
  local target_path="$3"
  local safe_name="$4"
  local source_manifest="$MANIFEST_SRC_DIR/${safe_name}.manifest"
  local target_manifest="$MANIFEST_TGT_DIR/${safe_name}.manifest"
  local diff_file="$DIFF_DIR/${safe_name}.diff"
  local source_hash
  local target_hash
  local status
  local details

  if [[ "$REUSE_MANIFESTS" != "0" && -f "$source_manifest" ]]; then
    echo "  - 复用源 manifest: $source_manifest"
  else
    echo "  - 生成源 manifest"
    build_manifest "$source_path" "$source_manifest"
  fi

  if [[ "$REUSE_MANIFESTS" != "0" && -f "$target_manifest" ]]; then
    echo "  - 复用目标 manifest: $target_manifest"
  else
    echo "  - 生成目标 manifest"
    build_manifest "$target_path" "$target_manifest"
  fi

  source_hash="$(manifest_hash "$source_manifest")"
  target_hash="$(manifest_hash "$target_manifest")"

  if [[ "$source_hash" == "$target_hash" ]]; then
    status="MATCH"
    details="manifest_hash 相同"
    rm -f "$diff_file"
  else
    status="MISMATCH"
    details="manifest_hash 不同，详见 diff"
    diff -u --label "source/$scope" --label "target/$scope" \
      "$source_manifest" "$target_manifest" > "$diff_file" || true
  fi

  write_summary "$scope" "dir" "$status" "$source_hash" "$target_hash" "$source_path" "$target_path" "$details"

  echo "  - source_hash: $source_hash"
  echo "  - target_hash: $target_hash"
  echo "  - status: $status"
  echo "$scope: $status" >> "$REPORT_TXT"
}

compare_file_scope() {
  local scope="$1"
  local source_path="$2"
  local target_path="$3"
  local item_type="$4"
  local source_hash
  local target_hash
  local status
  local details

  if [[ "$item_type" == "symlink" ]]; then
    source_hash="$(readlink -- "$source_path")"
    target_hash="$(readlink -- "$target_path")"
  else
    source_hash="$(sha256sum -- "$source_path" | awk '{print $1}')"
    target_hash="$(sha256sum -- "$target_path" | awk '{print $1}')"
  fi

  if [[ "$source_hash" == "$target_hash" ]]; then
    status="MATCH"
    details="内容相同"
  else
    status="MISMATCH"
    details="内容不同"
  fi

  write_summary "$scope" "$item_type" "$status" "$source_hash" "$target_hash" "$source_path" "$target_path" "$details"

  echo "  - source_hash: $source_hash"
  echo "  - target_hash: $target_hash"
  echo "  - status: $status"
  echo "$scope: $status" >> "$REPORT_TXT"
}

compare_pair() {
  local scope="$1"
  local source_path="$2"
  local target_path="$3"
  local safe_name="$4"
  local source_type
  local target_type

  source_type="$(item_type_of "$source_path")"
  target_type="$(item_type_of "$target_path")"

  echo "[$(timestamp)] 比较范围: $scope"

  if [[ "$source_type" != "$target_type" ]]; then
    write_summary "$scope" "$source_type->$target_type" "TYPE_MISMATCH" "-" "-" "$source_path" "$target_path" "同名项类型不同"
    echo "  - source_type: $source_type"
    echo "  - target_type: $target_type"
    echo "  - status: TYPE_MISMATCH"
    echo "$scope: TYPE_MISMATCH" >> "$REPORT_TXT"
    return
  fi

  case "$source_type" in
    dir)
      compare_dir_scope "$scope" "$source_path" "$target_path" "$safe_name"
      ;;
    file|symlink)
      compare_file_scope "$scope" "$source_path" "$target_path" "$source_type"
      ;;
    *)
      write_summary "$scope" "$source_type" "SKIPPED" "-" "-" "$source_path" "$target_path" "暂不支持的类型"
      echo "  - status: SKIPPED"
      echo "$scope: SKIPPED" >> "$REPORT_TXT"
      ;;
  esac
}

declare -A TARGET_NAMES=()
while IFS= read -r -d '' name; do
  TARGET_NAMES["$name"]=1
done < <(find "$TARGET_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\0')

common_backup_count=0
while IFS= read -r -d '' name; do
  if [[ -n "${TARGET_NAMES[$name]:-}" ]]; then
    ((common_backup_count += 1))
    compare_pair "BACKUP_TMP/$name" "$SOURCE_CONTENT_ROOT/$name" "$TARGET_ROOT/$name" "backup_${name}"
  fi
done < <(find "$SOURCE_CONTENT_ROOT" -mindepth 1 -maxdepth 1 -printf '%f\0' | sort -z)

common_root_file_count=0
while IFS= read -r -d '' name; do
  if [[ -n "${TARGET_NAMES[$name]:-}" ]]; then
    ((common_root_file_count += 1))
    compare_pair "raid0-root/$name" "$SOURCE_SHARE_ROOT/$name" "$TARGET_ROOT/$name" "root_${name}"
  fi
done < <(find "$SOURCE_SHARE_ROOT" -mindepth 1 -maxdepth 1 \( -type f -o -type l \) -printf '%f\0' | sort -z)

match_count="$(awk -F '\t' 'NR > 1 && $3 == "MATCH" {count++} END {print count + 0}' "$SUMMARY_TSV")"
mismatch_count="$(awk -F '\t' 'NR > 1 && $3 == "MISMATCH" {count++} END {print count + 0}' "$SUMMARY_TSV")"
type_mismatch_count="$(awk -F '\t' 'NR > 1 && $3 == "TYPE_MISMATCH" {count++} END {print count + 0}' "$SUMMARY_TSV")"
skipped_count="$(awk -F '\t' 'NR > 1 && $3 == "SKIPPED" {count++} END {print count + 0}' "$SUMMARY_TSV")"

{
  echo
  echo "Summary"
  echo "common_backup_items=$common_backup_count"
  echo "common_raid0_root_files=$common_root_file_count"
  echo "match=$match_count"
  echo "mismatch=$mismatch_count"
  echo "type_mismatch=$type_mismatch_count"
  echo "skipped=$skipped_count"
  echo "finished_at: $(timestamp)"
} | tee -a "$REPORT_TXT"

cat >> "$META_TXT" <<EOF
finished_at=$(timestamp)
common_backup_items=$common_backup_count
common_raid0_root_files=$common_root_file_count
match=$match_count
mismatch=$mismatch_count
type_mismatch=$type_mismatch_count
skipped=$skipped_count
EOF

echo
echo "结果文件:"
echo "  - $SUMMARY_TSV"
echo "  - $REPORT_TXT"
echo "  - $RUN_LOG"
