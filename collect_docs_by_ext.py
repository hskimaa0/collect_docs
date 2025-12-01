import os
import shutil
import hashlib
from pathlib import Path

# ========= 설정 부분 =========

# NAS 원본 폴더 (UNC 경로)
SOURCE_DIR = r"\\192.168.0.2\sol_nas"

# 내 로컬 PC에 모을 루트 폴더
DEST_ROOT = r"D:\collect_docs"

# 수집할 문서 확장자 목록 (소문자로 작성)
DOC_EXTENSIONS = {
    ".pdf",
    ".doc", ".docx",
    ".xls", ".xlsx",
    ".ppt", ".pptx",
    ".hwp", ".hwpx",
}

# 실제 복사 대신 "어디로 갈지"만 보고 싶으면 True로
DRY_RUN = False

# ============================

def ensure_dir(path: Path):
    """폴더 없으면 생성"""
    if not path.exists():
        print(f"[DIR] 만들기: {path}")
        path.mkdir(parents=True, exist_ok=True)

def get_file_hash(filepath: Path, chunk_size: int = 8192) -> str:
    """파일의 MD5 해시를 계산"""
    hasher = hashlib.md5()
    with open(filepath, 'rb') as f:
        while chunk := f.read(chunk_size):
            hasher.update(chunk)
    return hasher.hexdigest()

def collect_existing_hashes(dest_root: Path) -> set:
    """이미 복사된 파일들의 해시를 수집"""
    existing_hashes = set()
    if not dest_root.exists():
        return existing_hashes
    
    print("[INFO] 기존 파일들의 해시 수집 중...")
    count = 0
    for root, dirs, files in os.walk(dest_root):
        for name in files:
            filepath = Path(root) / name
            try:
                file_hash = get_file_hash(filepath)
                existing_hashes.add(file_hash)
                count += 1
            except Exception as e:
                print(f"[경고] 해시 계산 실패: {filepath} ({e})")
    print(f"[INFO] 기존 파일 {count}개의 해시 수집 완료")
    return existing_hashes

def main():
    src = Path(SOURCE_DIR)
    dest_root = Path(DEST_ROOT)

    if not src.exists():
        print(f"[에러] 원본 경로가 존재하지 않습니다: {src}")
        return

    ensure_dir(dest_root)

    # 이미 복사된 파일들의 해시 수집
    existing_hashes = collect_existing_hashes(dest_root)

    file_count = 0
    skip_count = 0

    # 재귀적으로 전체 탐색
    for root, dirs, files in os.walk(src):
        for name in files:
            src_path = Path(root) / name
            ext = src_path.suffix.lower()  # ".PDF" -> ".pdf"

            # 문서 확장자만 필터링
            if ext not in DOC_EXTENSIONS:
                continue

            # 확장자별 하위 폴더 이름 (".pdf" -> "pdf")
            folder_name = ext.lstrip(".") if ext else "no_extension"
            dest_folder = dest_root / folder_name
            ensure_dir(dest_folder)

            dest_path = dest_folder / src_path.name

            if DRY_RUN:
                print(f"[DRY_RUN] {src_path} -> {dest_path}")
            else:
                try:
                    # 파일 해시 계산
                    src_hash = get_file_hash(src_path)
                    
                    # 동일한 내용의 파일이 이미 있으면 건너뛰기
                    if src_hash in existing_hashes:
                        print(f"[SKIP] 동일한 내용의 파일 존재: {src_path.name}")
                        skip_count += 1
                        continue
                    
                    print(f"[COPY] {src_path} -> {dest_path}")
                    # 메타데이터까지 같이 복사 (수정시간 등)
                    shutil.copy2(src_path, dest_path)
                    existing_hashes.add(src_hash)  # 새로 복사한 파일의 해시도 추가
                    file_count += 1
                except Exception as e:
                    print(f"[에러] 복사 실패: {src_path} -> {dest_path} ({e})")

    if DRY_RUN:
        print("\n[완료] DRY_RUN 모드라 실제 복사는 하지 않았습니다.")
    else:
        print(f"\n[완료] 총 {file_count}개 파일 복사, {skip_count}개 파일 건너뜀 (동일한 파일).")

if __name__ == "__main__":
    main()
