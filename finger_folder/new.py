#!/usr/bin/env python
"""
fingerprint_binarize.py
-----------------------
카메라로 촬영한 손가락 지문 사진을
1) 대비 향상(CLAHE) → 2) 가보(Gabor) 기반 융선 강조 →
3) Sauvola 적응 이진화 → 4) 작은 틈 메우기 →
5) 자동 ROI(손가락 영역) 마스킹 → 6) 최종 바이너리 저장

필수 라이브러리
    pip install opencv-python scikit-image fingerprint_enhancer numpy
"""

from pathlib import Path
import argparse

import cv2
import numpy as np
import fingerprint_enhancer as fe
from skimage.filters import threshold_sauvola
from skimage.morphology import closing, square


# ---------------------------------------------------------------------------
# ① 융선 강조 – fingerprint_enhancer 버전별 함수명 호환
def _enhance(img: np.ndarray) -> np.ndarray:
    """
    fingerprint_enhancer 모듈에서 사용 가능한 enhance 함수 호출.
    (버전에 따라 enhance_fingerprint / enhance_Fingerprint 둘 중 하나만 있을 수 있음)
    """
    if hasattr(fe, "enhance_fingerprint"):
        return fe.enhance_fingerprint(img)
    if hasattr(fe, "enhance_Fingerprint"):
        return fe.enhance_Fingerprint(img)
    raise AttributeError(
        "fingerprint_enhancer 모듈에 enhance 함수가 없습니다."
        " pip 설치가 올바른지 확인하세요."
    )


# ---------------------------------------------------------------------------
# ② 손가락 ROI 마스크 생성
def _build_roi_mask(
    binary: np.ndarray,
    dilate_iter: int = 3,
    close_iter: int = 3,
    kernel_size: int = 7,
) -> np.ndarray:
    """
    융선 이진화 결과(binary)를 기반으로 손가락 영역만 남기는 ROI 마스크 생성.
    dilate → close → floodfill → 가장 큰 외곽선만 채우는 순서.
    Returns
    -------
    mask : np.ndarray
        uint8(0/255) 마스크. 손가락 영역=255, 배경=0
    """
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # 1) 팽창 – ridges를 굵게 만들어 끊김 최소화
    mask = cv2.dilate(binary, ker, iterations=dilate_iter)

    # 2) 클로징 – 내부 작은 구멍 메움
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, ker, iterations=close_iter)

    # 3) flood fill로 배경 영역을 흰색으로 채운 뒤 반전 → 손가락 내부 빈칸까지 채우기
    flood = mask.copy()
    h, w = mask.shape
    cv2.floodFill(flood, None, (0, 0), 255)          # 모서리(배경) floodfill
    mask = cv2.bitwise_not(flood) | mask             # 손가락 영역만 255

    # 4) 가장 큰 외곽선(손가락)만 마스크로 유지
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.zeros_like(mask)
    largest = max(contours, key=cv2.contourArea)
    roi = np.zeros_like(mask)
    cv2.drawContours(roi, [largest], -1, 255, thickness=-1)
    return roi


# ---------------------------------------------------------------------------
def binarize_fingerprint(
    img_path: str | Path,
    out_path: str | Path,
    *,
    window_size: int = 25,
    k: float = 0.2,
    fill_gaps: bool = True,
    debug: bool = False,
) -> np.ndarray:
    """
    손가락 지문 사진을 이진화 + ROI 마스킹 후 저장.

    Parameters
    ----------
    img_path : str | Path
        입력 원본 사진(컬러·그레이 모두 가능)
    out_path : str | Path
        저장 경로
    window_size : int
        Sauvola 윈도 크기
    k : float
        Sauvola k 파라미터
    fill_gaps : bool
        융선 사이 작은 틈 closing 으로 메울지
    debug : bool
        True면 중간 결과(강조 이미지·ROI 마스크) 추가 저장
    """
    img_path = Path(img_path)
    out_path = Path(out_path)

    # ---------- 원본 불러오기 ----------
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(img_path)

    # ---------- CLAHE 대비 향상 ----------
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    img_clahe = clahe.apply(img)

    # ---------- 융선 강조 ----------
    enhanced = _enhance(img_clahe)

    # ---------- Sauvola 적응 임계값 ----------
    thresh = threshold_sauvola(enhanced, window_size, k=k)
    binary = (enhanced > thresh).astype(np.uint8)

    # ---------- 작은 gap 메우기 ----------
    if fill_gaps:
        binary = closing(binary, square(3)).astype(np.uint8)

    binary *= 255  # 0/1 → 0/255 uint8

    # ---------- 자동 ROI 마스크 ----------
    mask = _build_roi_mask(binary)

    # ---------- 마스킹 & 저장 ----------
    final = cv2.bitwise_and(binary, mask)

    final = cv2.flip(final, 1)

    
    cv2.imwrite(str(out_path), final)

    if debug:
        cv2.imwrite(str(out_path.with_stem(out_path.stem + "_enhanced")), enhanced)
        cv2.imwrite(str(out_path.with_stem(out_path.stem + "_mask")), mask)

    return final


# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fingerprint photo → binary with automatic ROI mask"
    )
    parser.add_argument("input", help="원본 지문 사진 파일")
    parser.add_argument(
        "-o",
        "--output",
        default="fingerprint_binary.png",
        help="이진화 결과 저장 파일 (기본: fingerprint_binary.png)",
    )
    parser.add_argument(
        "--window-size", type=int, default=25, help="Sauvola window size (default 25)"
    )
    parser.add_argument(
        "-k", type=float, default=0.2, help="Sauvola k (default 0.2, 0.16~0.25 추천)"
    )
    parser.add_argument(
        "--no-fill-gaps",
        action="store_true",
        help="Morphological closing(틈 메우기) 끄기",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="중간 결과(강조·마스크)도 저장",
    )
    args = parser.parse_args()

    binarize_fingerprint(
        args.input,
        args.output,
        window_size=args.window_size,
        k=args.k,
        fill_gaps=not args.no_fill_gaps,
        debug=args.debug,
    )
    print(f"✔️  결과 저장 완료 → {args.output}")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    main()
