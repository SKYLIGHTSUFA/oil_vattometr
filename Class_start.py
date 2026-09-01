"""Окно оператора: выбрать папку с графиками и получить список обрывов.

Инференс уходит в отдельный поток — иначе окно замирает на всё время
предсказания и Windows рисует «программа не отвечает».
"""
from __future__ import annotations

import queue
import threading
from pathlib import Path

import config as cfg_mod
from Class_Predict import BeltPredictor, Prediction


class BeltApp:
    """Простой GUI поверх :class:`BeltPredictor`."""

    def __init__(self, weights: Path | str = cfg_mod.DEFAULT_WEIGHTS) -> None:
        import customtkinter as ctk

        self.ctk = ctk
        self.weights = Path(weights)
        self.predictor: BeltPredictor | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Обрыв ремней СК")
        self.root.geometry("760x520")
        self.root.grid_columnconfigure(0, weight=1)
        self.root.grid_rowconfigure(2, weight=1)

        top = ctk.CTkFrame(self.root)
        top.grid(row=0, column=0, padx=12, pady=12, sticky="ew")
        top.grid_columnconfigure(0, weight=1)

        self.path_entry = ctk.CTkEntry(top, placeholder_text="Папка с графиками")
        self.path_entry.grid(row=0, column=0, padx=(8, 8), pady=8, sticky="ew")
        ctk.CTkButton(top, text="Обзор", width=90, command=self.choose_dir).grid(row=0, column=1, padx=4, pady=8)
        self.run_button = ctk.CTkButton(top, text="Проверить", width=110, command=self.start)
        self.run_button.grid(row=0, column=2, padx=(4, 8), pady=8)

        self.status = ctk.CTkLabel(self.root, text="Выберите папку с графиками", anchor="w")
        self.status.grid(row=1, column=0, padx=16, sticky="ew")

        self.output = ctk.CTkTextbox(self.root)
        self.output.grid(row=2, column=0, padx=12, pady=12, sticky="nsew")

        self.root.after(100, self._drain_events)

    # ------------------------------------------------------------------ #
    def choose_dir(self) -> None:
        from tkinter import filedialog

        chosen = filedialog.askdirectory(title="Папка с графиками")
        if chosen:
            self.path_entry.delete(0, "end")
            self.path_entry.insert(0, chosen)

    def start(self) -> None:
        source = self.path_entry.get().strip()
        if not source:
            self._set_status("Сначала укажите папку")
            return

        self.run_button.configure(state="disabled")
        self.output.delete("1.0", "end")
        self._set_status("Считаю...")
        threading.Thread(target=self._work, args=(Path(source),), daemon=True).start()

    def _work(self, source: Path) -> None:
        """Фоновый поток: тут нельзя трогать виджеты, только класть события в очередь."""
        try:
            if self.predictor is None:
                self.predictor = BeltPredictor(self.weights)
            self.events.put(("done", self.predictor.predict(source)))
        except Exception as exc:  # noqa: BLE001 — любое падение показываем оператору
            self.events.put(("error", exc))

    def _drain_events(self) -> None:
        while not self.events.empty():
            kind, payload = self.events.get()
            if kind == "done":
                self._show(list(payload))  # type: ignore[arg-type]
            else:
                self._set_status(f"Ошибка: {payload}")
            self.run_button.configure(state="normal")
        self.root.after(100, self._drain_events)

    def _show(self, predictions: list[Prediction]) -> None:
        if not predictions:
            self._set_status("Картинок не найдено")
            return

        bad = [p for p in predictions if p.is_broken]
        bad.sort(key=lambda p: p.confidence, reverse=True)
        for pred in bad:
            self.output.insert("end", f"{pred.confidence:6.1%}  {pred.path.name}\n")
        if not bad:
            self.output.insert("end", "Обрывов не найдено\n")
        self._set_status(f"Проверено {len(predictions)}, подозрений на обрыв: {len(bad)}")

    def _set_status(self, text: str) -> None:
        self.status.configure(text=text)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    BeltApp().run()


if __name__ == "__main__":
    main()
