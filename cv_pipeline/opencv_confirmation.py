#!/usr/bin/env python3
"""Native OpenCV confirmation for the compact LEGO showcase workspace."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from pipeline.inventory_constraints import InventoryCatalog, normalize_piece_name
from pipeline.showcase_handoff import write_showcase_handoff
from training.inventory_review import apply_component_review, atomic_write_json


WINDOW_NAME = "LEGO Scanner"
YELLOW = (0, 213, 255)
PALE_YELLOW = (178, 247, 255)
INK = (14, 18, 20)
WHITE = (244, 248, 252)
UP_KEYS = {2490368, 63232, 82}
DOWN_KEYS = {2621440, 63233, 84}


class ConfirmationController:
    def __init__(
        self,
        workspace_path: Path,
        catalog: InventoryCatalog,
        *,
        canvas_size: tuple[int, int] = (1280, 720),
        mascot_path: Path | None = None,
    ) -> None:
        self.workspace_path = Path(workspace_path).resolve()
        self.catalog = catalog
        self.canvas_size = tuple(map(int, canvas_size))
        workspace = json.loads(self.workspace_path.read_text())
        sessions = workspace.get("sessions") or []
        if not sessions:
            raise ValueError("confirmation workspace contains no sessions")
        review_ref = Path(sessions[0]["review_path"])
        self.review_path = (
            review_ref if review_ref.is_absolute()
            else (self.workspace_path.parent / review_ref)
        ).resolve()
        self.document = json.loads(self.review_path.read_text())
        if not self.document.get("showcase_only"):
            raise ValueError("native confirmation requires a showcase workspace")
        self.index = 0
        self.mode = "review"
        self.correction_text = ""
        self.suggestions: list[str] = []
        self.suggestion_index = 0
        self.status = ""
        self.button_rects: dict[str, tuple[int, int, int, int]] = {}
        self.quit_requested = False
        self.handoff: dict | None = None
        self.mascot_path = mascot_path or (
            Path(__file__).resolve().parent
            / "inventory_review_web/confirmation-mascot.png"
        )
        self._mascot = cv2.imread(str(self.mascot_path), cv2.IMREAD_UNCHANGED)
        self._select_next_unresolved()

    @property
    def components(self) -> list[dict]:
        return self.document.get("components", [])

    @property
    def current(self) -> dict:
        return self.components[self.index]

    @property
    def complete(self) -> bool:
        return self.mode == "complete"

    def _select_next_unresolved(self) -> None:
        for index, row in enumerate(self.components):
            if not (row.get("review") or {}).get("confirmed"):
                self.index = index
                self.mode = "review"
                return
        if not self.components:
            self.mode = "error"
            self.status = "No inventory-matched pieces were available."
            return
        self.handoff = write_showcase_handoff(self.review_path, self.document)
        self.mode = "complete"

    def _write_review(self, patch: dict) -> None:
        apply_component_review(
            self.document, self.current["component_id"], patch
        )
        atomic_write_json(self.review_path, self.document)
        self.status = "Saved"
        self.correction_text = ""
        self.suggestions = []
        self.suggestion_index = 0
        self._select_next_unresolved()

    def accept_prediction(self) -> bool:
        prediction = self.current.get("prediction", {})
        canonical = self.catalog.canonical_name(prediction.get("part_name"))
        if canonical is None:
            self.status = "Prediction is not present in the inventory."
            return False
        self._write_review({
            "confirmed": True,
            "part_id": prediction.get("part_id") or "",
            "part_name": canonical,
            "color": prediction.get("color") or "",
            "included": True,
            "exclusion_reason": None,
            "segmentation": "correct",
        })
        return True

    def begin_correction(self) -> None:
        self.mode = "correction"
        self.correction_text = ""
        self.status = "Type a piece from your inventory."
        self._refresh_suggestions()

    def _refresh_suggestions(self) -> None:
        query = normalize_piece_name(self.correction_text)
        self.suggestions = [
            name for name in self.catalog.allowed_names
            if not query or query in normalize_piece_name(name)
        ][:5]
        self.suggestion_index = min(
            self.suggestion_index, max(0, len(self.suggestions) - 1)
        )

    def save_current(self) -> bool:
        canonical = self.catalog.canonical_name(self.correction_text)
        if canonical is None and self.correction_text.strip() and self.suggestions:
            canonical = self.suggestions[self.suggestion_index]
        if canonical is None:
            self.status = "Choose a piece type that exists in your inventory."
            return False
        self._write_review({
            "confirmed": True,
            "part_id": "",
            "part_name": canonical,
            "color": self.current.get("prediction", {}).get("color") or "",
            "included": True,
            "exclusion_reason": None,
            "segmentation": "correct",
        })
        return True

    def handle_key(self, key: int) -> None:
        if key < 0:
            return
        low = key & 0xFF
        if self.mode in {"complete", "error"}:
            if low in {13, 27, ord("q"), ord("Q")}:
                self.quit_requested = True
            return
        if self.mode == "review":
            if low in {ord("y"), ord("Y")}:
                self.accept_prediction()
            elif low in {ord("n"), ord("N")}:
                self.begin_correction()
            elif low in {ord("q"), ord("Q"), 27}:
                self.quit_requested = True
            return
        if key in UP_KEYS:
            if self.suggestions:
                self.suggestion_index = max(0, self.suggestion_index - 1)
        elif key in DOWN_KEYS:
            if self.suggestions:
                self.suggestion_index = min(
                    len(self.suggestions) - 1, self.suggestion_index + 1
                )
        elif low == 27:
            self.mode = "review"
            self.status = ""
        elif low in {8, 127}:
            self.correction_text = self.correction_text[:-1]
            self._refresh_suggestions()
        elif low == 9 and self.suggestions:
            self.correction_text = self.suggestions[self.suggestion_index]
            self._refresh_suggestions()
        elif low in {10, 13}:
            self.save_current()
        elif 32 <= low <= 126:
            self.correction_text += chr(low)
            self._refresh_suggestions()

    def handle_click(self, x: int, y: int) -> None:
        for name, (x0, y0, x1, y1) in self.button_rects.items():
            if x0 <= x <= x1 and y0 <= y <= y1:
                if name == "yes":
                    self.accept_prediction()
                elif name == "no":
                    self.begin_correction()
                elif name == "back":
                    self.mode = "review"
                    self.status = ""
                elif name == "save":
                    self.save_current()
                elif name.startswith("suggestion-"):
                    self.suggestion_index = int(name.split("-")[-1])
                    self.correction_text = self.suggestions[self.suggestion_index]
                return

    def _frame_and_mask(self) -> tuple[np.ndarray, np.ndarray, list[int]]:
        frame_id = int(self.current["source_anchor"]["frame_id"])
        frame = next(
            row for row in self.document.get("source_frames", [])
            if int(row["frame_id"]) == frame_id
        )
        image_path = (self.review_path.parent / frame["image"]).resolve()
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"missing confirmation frame: {image_path}")
        view = (self.current.get("assets", {}).get("views") or [])[0]
        mask_path = (self.review_path.parent / view["mask"]).resolve()
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            raise ValueError(f"missing confirmation mask: {mask_path}")
        box = list(map(int, view.get("crop_box_original")
                       or self.current["source_anchor"]["box_original"]))
        return image, mask, box

    @staticmethod
    def _button(
        canvas: np.ndarray, rect: tuple[int, int, int, int], label: str,
        *, primary: bool = False,
    ) -> None:
        x0, y0, x1, y1 = rect
        fill = YELLOW if primary else (60, 64, 65)
        text_color = INK if primary else WHITE
        cv2.rectangle(canvas, (x0, y0), (x1, y1), fill, -1, cv2.LINE_AA)
        cv2.rectangle(canvas, (x0, y0), (x1, y1), YELLOW, 1, cv2.LINE_AA)
        size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.putText(
            canvas, label,
            (x0 + (x1 - x0 - size[0]) // 2, y0 + (y1 - y0 + size[1]) // 2),
            cv2.FONT_HERSHEY_SIMPLEX, 0.55, text_color, 2, cv2.LINE_AA,
        )

    def _draw_mascot(self, canvas: np.ndarray, left: int, top: int) -> None:
        if self._mascot is None or self._mascot.ndim != 3:
            return
        mascot = self._mascot
        width = 100
        height = max(1, round(mascot.shape[0] * width / mascot.shape[1]))
        mascot = cv2.resize(mascot, (width, height), interpolation=cv2.INTER_AREA)
        if top + height > canvas.shape[0] or left + width > canvas.shape[1]:
            return
        roi = canvas[top:top + height, left:left + width]
        if mascot.shape[2] == 4:
            alpha = mascot[:, :, 3:4].astype(np.float32) / 255.0
            roi[:] = (
                mascot[:, :, :3].astype(np.float32) * alpha
                + roi.astype(np.float32) * (1.0 - alpha)
            ).astype(np.uint8)
        else:
            roi[:] = mascot[:, :, :3]

    def render(self) -> np.ndarray:
        width, height = self.canvas_size
        canvas = np.full((height, width, 3), (0, 0, 0), dtype=np.uint8)
        self.button_rects = {}
        if self.mode in {"complete", "error"}:
            title = "Inventory ready" if self.mode == "complete" else "Confirmation unavailable"
            subtitle = (
                "Your confirmed pieces are ready for building."
                if self.mode == "complete" else self.status
            )
            cv2.putText(canvas, title, (80, height // 2 - 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, YELLOW, 3, cv2.LINE_AA)
            cv2.putText(canvas, subtitle, (82, height // 2 + 35),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, WHITE, 2, cv2.LINE_AA)
            return canvas

        image, mask, box = self._frame_and_mask()
        x0, y0, x1, y1 = box
        crop_h, crop_w = max(1, y1 - y0), max(1, x1 - x0)
        if mask.shape != (crop_h, crop_w):
            mask = cv2.resize(mask, (crop_w, crop_h), interpolation=cv2.INTER_NEAREST)
        x0, y0 = max(0, x0), max(0, y0)
        x1, y1 = min(image.shape[1], x1), min(image.shape[0], y1)
        usable = mask[:y1 - y0, :x1 - x0] > 0
        roi = image[y0:y1, x0:x1]
        yellow = np.empty_like(roi)
        yellow[:] = YELLOW
        blended = cv2.addWeighted(roi, 0.45, yellow, 0.55, 0)
        roi[usable] = blended[usable]

        scale = min(width / image.shape[1], height / image.shape[0])
        draw_w = max(1, round(image.shape[1] * scale))
        draw_h = max(1, round(image.shape[0] * scale))
        offset_x, offset_y = (width - draw_w) // 2, (height - draw_h) // 2
        canvas[offset_y:offset_y + draw_h, offset_x:offset_x + draw_w] = (
            cv2.resize(image, (draw_w, draw_h), interpolation=cv2.INTER_AREA)
        )

        card_w = min(350, width - 32)
        card_x, card_y = width - card_w - 18, 18
        card_h = 190 if self.mode == "review" else min(430, height - 36)
        overlay = canvas.copy()
        cv2.rectangle(
            overlay, (card_x, card_y), (card_x + card_w, card_y + card_h),
            (12, 18, 18), -1, cv2.LINE_AA,
        )
        cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
        cv2.rectangle(
            canvas, (card_x, card_y), (card_x + card_w, card_y + card_h),
            YELLOW, 2, cv2.LINE_AA,
        )
        center = (
            offset_x + round(((x0 + x1) / 2) * scale),
            offset_y + round(((y0 + y1) / 2) * scale),
        )
        cv2.line(canvas, center, (card_x, card_y + 85), YELLOW, 3, cv2.LINE_AA)
        cv2.circle(canvas, center, 7, PALE_YELLOW, -1, cv2.LINE_AA)
        cv2.circle(canvas, center, 10, YELLOW, 2, cv2.LINE_AA)

        cv2.putText(canvas, "QUICK CONFIRMATION", (card_x + 18, card_y + 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, YELLOW, 2, cv2.LINE_AA)
        if self.mode == "review":
            prediction = self.current["prediction"]
            question = f"Is this {prediction.get('part_name', 'LEGO piece')}?"
            cv2.putText(canvas, question, (card_x + 18, card_y + 62),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2, cv2.LINE_AA)
            confidence = round(float(prediction.get("score") or 0) * 100)
            cv2.putText(canvas, f"Model confidence {confidence}%",
                        (card_x + 18, card_y + 88),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (190, 205, 215), 1,
                        cv2.LINE_AA)
            yes = (card_x + 18, card_y + 116, card_x + 104, card_y + 158)
            no = (card_x + 116, card_y + 116, card_x + 196, card_y + 158)
            self.button_rects.update({"yes": yes, "no": no})
            self._button(canvas, yes, "Yes", primary=True)
            self._button(canvas, no, "No")
            self._draw_mascot(canvas, card_x + card_w - 112, card_y + card_h + 8)
        else:
            cv2.putText(canvas, "What LEGO piece is this?",
                        (card_x + 18, card_y + 61),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.58, WHITE, 2, cv2.LINE_AA)
            field = (card_x + 18, card_y + 78, card_x + card_w - 18, card_y + 118)
            cv2.rectangle(canvas, field[:2], field[2:], (245, 247, 249), -1)
            shown = self.correction_text[-28:] or "e.g. Brick 2 x 8"
            color = INK if self.correction_text else (120, 125, 130)
            cv2.putText(canvas, shown, (field[0] + 9, field[1] + 26),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.47, color, 1, cv2.LINE_AA)
            row_y = card_y + 137
            for index, suggestion in enumerate(self.suggestions):
                rect = (card_x + 18, row_y, card_x + card_w - 18, row_y + 34)
                if index == self.suggestion_index:
                    cv2.rectangle(canvas, rect[:2], rect[2:], (66, 72, 73), -1)
                cv2.putText(canvas, suggestion, (rect[0] + 7, rect[1] + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.43, WHITE, 1, cv2.LINE_AA)
                self.button_rects[f"suggestion-{index}"] = rect
                row_y += 36
            back = (card_x + 18, card_y + card_h - 52, card_x + 104, card_y + card_h - 14)
            save = (card_x + 116, card_y + card_h - 52, card_x + 214, card_y + card_h - 14)
            self.button_rects.update({"back": back, "save": save})
            self._button(canvas, back, "Back")
            self._button(canvas, save, "Save", primary=True)
        if self.status:
            cv2.putText(canvas, self.status[:48], (20, height - 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, PALE_YELLOW, 1,
                        cv2.LINE_AA)
        return canvas


def run_native_confirmation(
    workspace: Path, inventory_csv: Path, *, window_name: str = WINDOW_NAME
) -> dict | None:
    controller = ConfirmationController(
        workspace, InventoryCatalog.load(inventory_csv)
    )
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, *controller.canvas_size)

    def on_mouse(event, x, y, _flags, _param):
        if event == cv2.EVENT_LBUTTONUP:
            controller.handle_click(x, y)

    cv2.setMouseCallback(window_name, on_mouse)
    try:
        while not controller.quit_requested:
            cv2.imshow(window_name, controller.render())
            key = cv2.waitKeyEx(20)
            if key != -1:
                controller.handle_key(key)
            if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                break
    finally:
        cv2.destroyAllWindows()
    return controller.handoff


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--inventory-csv", type=Path, required=True)
    args = parser.parse_args(argv)
    result = run_native_confirmation(args.workspace, args.inventory_csv)
    if result:
        print(f"HANDOFF_READY {result['handoff_path']}", flush=True)


if __name__ == "__main__":
    main()
