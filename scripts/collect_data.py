"""허공 손글씨 데이터 수집기.

실행:
    python scripts/collect_data.py --note "책상 앞, 밝음"
    python scripts/collect_data.py --note "확인용" --holdout     # 학습에 안 쓰는 확인용 세션

화면 오른쪽 위에 '써야 할 기호'가 나옵니다. 그 기호를 허공에 쓰고 손바닥을 펴면 저장되고 다음 기호로 넘어갑니다.
라벨은 화면이 정해 주므로 키를 잘못 눌러 라벨이 틀어질 일이 없습니다.

손 모양
    검지만 펴기      쓰기
    검지+중지 (V)    이동 (획 사이, 예: + 의 가로를 긋고 세로 시작점으로 이동)
    손바닥 펴기      완료 -> 저장
    주먹 잠깐 유지   쓰던 글자 지우기

키보드
    space  완료 (손바닥 대신)       c  쓰던 글자 지우기
    r      방금 저장한 것 삭제 후 다시 쓰기
    s      이 기호 건너뛰기 (맨 뒤로)  q  종료

저장: data/sessions/<세션>/<기호>/<기호>_<ms>.npz (궤적 좌표만, 사진 없음) + 같은 이름의 .png (궤적 그림)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # 프로젝트 맨 위를 import 경로에 추가

import argparse
import json
import platform
import random
import re
import time
from datetime import datetime

import cv2

from aircalc.config import (
    ASCII,
    CLASSES,
    HOLDOUT_DIR,
    ROUNDS_PER_SESSION,
    SESSIONS_DIR,
    SYMBOL,
    WRITING_GUIDE,
)
from aircalc.drawing import FaceMosaic, draw_hand, draw_status_bar, draw_strokes, trajectory_preview
from aircalc.hand_tracking import HandTracker, open_camera
from aircalc.strokes import FEATURE_VERSION, AirWriter, save_sample

MIN_POINTS = 5
MAX_STROKES = 6


def build_queue(rounds, seed):
    rng = random.Random(seed)
    queue = []
    for _ in range(rounds):
        order = CLASSES[:]
        rng.shuffle(order)
        queue += order
    return queue


def announce(target, done, total):
    guide = WRITING_GUIDE.get(target, "")
    print(f"[{done + 1}/{total}] 쓸 기호: {SYMBOL[target]}" + (f"   ({guide})" if guide else ""))


def main():
    parser = argparse.ArgumentParser(description="허공 손글씨 데이터 수집")
    parser.add_argument("--subject", default="me", help="쓰는 사람 ID (혼자면 기본값 그대로)")
    parser.add_argument("--note", default="", help="촬영 조건 메모")
    parser.add_argument("--rounds", type=int, default=ROUNDS_PER_SESSION, help="15개 기호를 몇 바퀴 쓸지")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--holdout", action="store_true", help="확인용 세션으로 저장 (학습에 사용 안 함)")
    args = parser.parse_args()

    subject = re.sub(r"[^\w-]", "_", args.subject.strip()) or "me"
    session_id = f"{datetime.now():%Y%m%d_%H%M%S}_{subject}"
    session_dir = (HOLDOUT_DIR if args.holdout else SESSIONS_DIR) / session_id
    session_dir.mkdir(parents=True, exist_ok=True)

    tracker = HandTracker()
    try:
        cap, (width, height) = open_camera(args.camera)
    except Exception:
        tracker.close()
        raise
    aspect = width / height
    writer, face = AirWriter(), FaceMosaic()

    queue = build_queue(args.rounds, seed=int(time.time()))
    total = len(queue)
    counts = {c: 0 for c in CLASSES}
    saved = []                                  # (path, label)
    info = {"session_id": session_id, "subject": subject, "split": "holdout" if args.holdout else "train",
            "note": args.note, "platform": platform.platform(), "frame_size": [width, height],
            "feature_version": FEATURE_VERSION, "created_at": datetime.now().isoformat(timespec="seconds"),
            "saved_samples": counts}
    info_path = session_dir / "session_info.json"

    def write_info():
        info["total_samples"] = sum(counts.values())
        info_path.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    def save(strokes, label):
        n_points = sum(len(s) for s in strokes)
        if n_points < MIN_POINTS:
            return f"too short ({n_points} pts) - write again"
        if len(strokes) > MAX_STROKES:
            return f"too many strokes ({len(strokes)}) - write again"
        path = session_dir / label / f"{label}_{int(time.time() * 1000)}.npz"
        path.parent.mkdir(exist_ok=True)
        save_sample(path, strokes, label, aspect)
        cv2.imwrite(str(path.with_suffix(".png")), trajectory_preview(strokes))
        saved.append((path, label))
        counts[label] += 1
        write_info()
        queue.pop(0)
        print(f"   저장: {SYMBOL[label]} (획 {len(strokes)}개, 점 {n_points}개)")
        if queue:
            announce(queue[0], total - len(queue), total)
        return f"saved {ASCII[label]}"

    write_info()
    print("=" * 64)
    print(f"세션 {session_id}{'  [확인용]' if args.holdout else ''} | 총 {total}개 ({len(CLASSES)}개 기호 x {args.rounds}바퀴)")
    print("검지만 펴서 쓰기 / V손으로 이동 / 손바닥 펴서 저장 / 주먹으로 지우기")
    print("키: space=저장  c=지우기  r=방금 저장 삭제  s=건너뛰기  q=종료")
    print("=" * 64)
    announce(queue[0], 0, total)
    message = ""

    try:
        while queue:
            ok, frame = cap.read()
            if not ok:
                print("웹캠 프레임을 읽지 못했습니다.")
                break
            frame = cv2.flip(frame, 1)
            now = time.perf_counter()
            hand = tracker.process(frame)
            event = writer.update(now, hand, aspect)
            target = queue[0]

            if event and event[0] == "confirm":
                message = save(event[1], target)
            elif event and event[0] == "clear":
                message = "cleared"

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                strokes = writer.force_confirm()
                message = save(strokes, target) if strokes else "nothing to save"
            elif key == ord("c"):
                writer.clear()
                message = "cleared"
            elif key == ord("s"):
                queue.append(queue.pop(0))
                writer.clear()
                message = f"skipped {ASCII[target]}"
                announce(queue[0], total - len(queue), total)
            elif key == ord("r") and saved:
                path, label = saved.pop()
                path.unlink(missing_ok=True)
                path.with_suffix(".png").unlink(missing_ok=True)
                counts[label] -= 1
                write_info()
                queue.insert(0, label)
                writer.clear()
                message = f"deleted last ({ASCII[label]}) - write it again"
                print(f"   삭제: {path.name}")
                announce(queue[0], total - len(queue), total)

            # ---------------- 화면 ----------------
            view = face(frame.copy())
            draw_strokes(view, writer.visible_strokes(), aspect)
            draw_hand(view, hand, writer.gesture)
            draw_status_bar(view, writer.gesture, f"{total - len(queue)}/{total}")
            if queue:
                h, w = view.shape[:2]
                sym = ASCII[queue[0]]
                cv2.rectangle(view, (w - 130, 50), (w - 10, 170), (30, 30, 30), -1)
                cv2.putText(view, "WRITE", (w - 112, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
                cv2.putText(view, sym, (w - 98, 150), cv2.FONT_HERSHEY_SIMPLEX, 2.6, (40, 230, 255), 5, cv2.LINE_AA)
                if queue[0] in WRITING_GUIDE:
                    cv2.putText(view, queue[0], (w - 125, 190), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 230, 255), 1,
                                cv2.LINE_AA)
            if message:
                cv2.putText(view, message, (10, view.shape[0] - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2,
                            cv2.LINE_AA)
            cv2.imshow("Air Writing - Data Collection", view)
    finally:
        cap.release()
        cv2.destroyAllWindows()
        tracker.close()
        info["ended_at"] = datetime.now().isoformat(timespec="seconds")
        write_info()

    print("\n" + "=" * 64)
    print("  ".join(f"{SYMBOL[c]}:{n}" for c, n in counts.items()))
    print(f"총 {sum(counts.values())}개 저장 -> {session_dir}")
    print("=" * 64)


if __name__ == "__main__":
    main()
