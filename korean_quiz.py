from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import platform
import random
import shutil
import subprocess
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from datetime import datetime

import string

PUNCT_TRANSLATOR = str.maketrans("", "", string.punctuation)

import edge_tts
import tkinter as tk
from tkinter import messagebox

Vocabulary = List[Tuple[str, str]]
History = List[int]
WordRecord = Dict[str, History]
CategoryStats = Dict[str, WordRecord]
Progress = Dict[str, CategoryStats]


def load_vocab(path: Path) -> Vocabulary:
    """Return (Korean, English) pairs loaded from a JSON file."""
    with path.open(encoding="utf-8-sig") as handle:
        raw_items = json.load(handle)

    vocabulary: Vocabulary = []
    for item in raw_items:
        if not isinstance(item, dict):
            raise ValueError(f"{path} contains an invalid entry (expected an object).")
        try:
            korean = str(item["korean"])
            english = str(item["english"])
        except KeyError as exc:
            raise ValueError(f"{path} entries must include 'korean' and 'english'.") from exc
        vocabulary.append((korean, english))

    if len(vocabulary) < 4:
        raise ValueError(f"{path} must contain at least four entries.")

    unique_translations = {english for _, english in vocabulary}
    if len(unique_translations) < 4:
        raise ValueError(f"{path} must contain at least four unique English translations.")
    unique_korean = {korean for korean, _ in vocabulary}
    if len(unique_korean) < 4:
        raise ValueError(f"{path} must contain at least four unique Korean entries.")

    return vocabulary


@dataclass
class Question:
    prompt: str
    options: List[str]
    answer: str
    korean: str


@dataclass(frozen=True)
class CategoryDefinition:
    name: str
    path: Path
    group: str


DEFAULT_CATALOG = {
    "groups": [
        {
            "name": "Lessons",
            "categories": [
                {"name": "Class Lesson", "file": "class_lesson_story.json"},
            ],
        },
        {
            "name": "Vocabulary & Grammar",
            "categories": [
                {"name": "Sino Numbers", "file": "sino_numbers.json"},
                {"name": "Native Numbers", "file": "native_numbers.json"},
                {"name": "Particles", "file": "particles.json"},
                {"name": "Nouns", "file": "nouns.json"},
                {"name": "Verbs", "file": "verbs.json"},
                {"name": "Adjectives", "file": "adjectives.json"},
                {"name": "Adverbs", "file": "adverbs.json"},
                {"name": "Interrogatives", "file": "interrogatives.json"},
                {"name": "Sentences", "file": "sentences.json"},
                {"name": "SKNS1b Vocabulary 1", "file": "SKNS1b_vocabulary1.json"},
                {"name": "SKNS1b Vocabulary 2", "file": "SKNS1b_vocabulary2.json"},
                {"name": "SKNS1b Vocabulary 3", "file": "SKNS1b_vocabulary3.json"},
                {"name": "SKNS1b Vocabulary 4", "file": "SKNS1b_vocabulary4.json"},
                {"name": "SKNS1b Vocabulary 5", "file": "SKNS1b_vocabulary5.json"},
            ],
        },
    ]
}


def make_question(pool: Vocabulary, selected: Tuple[str, str], direction: str) -> Question:
    """Create a multiple-choice question using a pre-selected vocabulary item."""
    korean, english = selected
    if direction == "ko-en":
        distractors = [candidate for _, candidate in pool if candidate != english]
        if len(distractors) < 3:
            raise ValueError("Vocabulary must include at least four unique meanings.")
        options = random.sample(distractors, k=3)
        options.append(english)
        random.shuffle(options)
        prompt = korean
        answer = english
    else:
        distractors = [candidate for candidate, _ in pool if candidate != korean]
        if len(distractors) < 3:
            raise ValueError("Vocabulary must include at least four unique Korean words.")
        options = random.sample(distractors, k=3)
        options.append(korean)
        random.shuffle(options)
        prompt = english
        answer = korean
    return Question(prompt=prompt, options=options, answer=answer, korean=korean)


def compute_weight(history: History, mode: str) -> float:
    """Return a sampling weight that favours new and troublesome items."""
    attempts = len(history)
    wrong = history.count(0)
    wrong_ratio = (wrong / attempts) if attempts else 1.0

    if attempts < 10:
        fill = 10 - attempts
        if mode == "fresh":
            weight = 18.0 + fill * 8.0 + wrong_ratio * 4.0
        else:
            weight = 10.0 + fill * 5.0 + wrong_ratio * 8.0
    else:
        if wrong_ratio == 0:
            weight = 0.25 if mode == "fresh" else 0.1
        else:
            base = 1.5 if mode == "fresh" else 1.0
            multiplier = 8.0 if mode == "fresh" else 14.0
            weight = base + wrong_ratio * multiplier

    return max(weight, 0.1)


def build_question(pool: Vocabulary, stats: CategoryStats, mode: str, direction: str) -> Question:
    """Select a vocabulary item using weighted sampling and build a question."""
    weights: List[float] = []
    for korean, english in pool:
        record = stats.get(korean)
        history = list(record.get("history", [])) if record else []
        weight = compute_weight(history, mode)
        weights.append(weight)
    selected = random.choices(pool, weights=weights, k=1)[0]
    return make_question(pool, selected, direction)


def normalize_answer(text: str) -> str:
    """Normalize user input for comparison."""
    normalized = text.strip().lower()
    normalized = normalized.translate(PUNCT_TRANSLATOR)
    normalized = " ".join(normalized.split())
    return normalized


def load_progress(path: Path, categories: List[str]) -> Tuple[Progress, Dict[str, Any]]:
    """Load per-word progress (history of last answers) and metadata from disk."""
    progress: Progress = {category: {} for category in categories}
    metadata: Dict[str, Any] = {
        "streak": 0,
        "streak_timestamp": "",
        "longest_streak": 0,
        "longest_streak_timestamp": "",
    }
    if not path.exists():
        return progress, metadata

    with path.open(encoding="utf-8-sig") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("Progress file must contain a JSON object.")

    raw_meta = data.get("__meta__")
    if isinstance(raw_meta, dict):
        try:
            metadata["streak"] = max(0, int(raw_meta.get("streak", metadata["streak"])))
        except (TypeError, ValueError):
            pass
        metadata["streak_timestamp"] = str(
            raw_meta.get("streak_timestamp") or metadata["streak_timestamp"]
        )
        try:
            metadata["longest_streak"] = max(
                0, int(raw_meta.get("longest_streak", metadata["longest_streak"]))
            )
        except (TypeError, ValueError):
            pass
        metadata["longest_streak_timestamp"] = str(
            raw_meta.get("longest_streak_timestamp")
            or metadata["longest_streak_timestamp"]
        )

    for category in categories:
        raw_stats = data.get(category, {})
        if not isinstance(raw_stats, dict):
            continue
        category_stats: CategoryStats = {}
        for korean, record in raw_stats.items():
            history: History = []

            if isinstance(record, dict):
                if isinstance(record.get("history"), list):
                    history = [
                        1 if bool(value) else 0
                        for value in record["history"]
                        if isinstance(value, (int, bool))
                    ]
                else:
                    attempts = record.get("attempts", 0)
                    correct = record.get("correct", 0)
                    try:
                        attempts_int = int(attempts)
                        correct_int = int(correct)
                    except (TypeError, ValueError):
                        attempts_int = 0
                        correct_int = 0
                    if attempts_int > 0:
                        correct_int = max(0, min(correct_int, attempts_int))
                        wrong = attempts_int - correct_int
                        history = [1] * min(correct_int, 10)
                        history.extend([0] * min(wrong, 10 - len(history)))
            elif isinstance(record, list):
                history = [1 if bool(value) else 0 for value in record if isinstance(value, (int, bool))]

            history = history[-10:]
            if history:
                category_stats[str(korean)] = {"history": history}
        progress[category] = category_stats
    return progress, metadata


def load_category_catalog(
    data_dir: Path, catalog_path: Path
) -> Tuple[List[CategoryDefinition], List[Tuple[str, List[str]]]]:
    """Load grouped category definitions from disk (or fall back to defaults)."""

    def _derive_name(file_token: str) -> str:
        stem = Path(file_token).stem.replace("_", " ").strip()
        return stem.title() if stem else "Untitled"

    if catalog_path.exists():
        with catalog_path.open(encoding="utf-8-sig") as handle:
            raw_catalog = json.load(handle)
    else:
        raw_catalog = DEFAULT_CATALOG
        try:
            catalog_path.parent.mkdir(parents=True, exist_ok=True)
            with catalog_path.open("w", encoding="utf-8") as handle:
                json.dump(raw_catalog, handle, ensure_ascii=False, indent=2)
        except OSError:
            # Fall back to the in-memory default catalog if the file cannot be written.
            pass

    if isinstance(raw_catalog, dict) and "groups" in raw_catalog:
        raw_groups = raw_catalog["groups"]
    elif isinstance(raw_catalog, dict):
        raw_groups = [
            {"name": str(group_name), "categories": entries}
            for group_name, entries in raw_catalog.items()
        ]
    else:
        raise ValueError("Catalog must be a JSON object.")

    if not isinstance(raw_groups, list) or not raw_groups:
        raise ValueError("Catalog 'groups' entry must be a non-empty list.")

    definitions: List[CategoryDefinition] = []
    grouped_names: List[Tuple[str, List[str]]] = []
    seen_names: set[str] = set()

    for raw_group in raw_groups:
        if not isinstance(raw_group, dict):
            raise ValueError("Each catalog group must be an object.")
        group_name = str(raw_group.get("name") or "").strip()
        if not group_name:
            raise ValueError("Each catalog group requires a non-empty 'name'.")
        entries = raw_group.get("categories")
        if not isinstance(entries, list) or not entries:
            continue
        group_categories: List[str] = []
        for entry in entries:
            if isinstance(entry, str):
                file_token = entry.strip()
                if not file_token:
                    raise ValueError("Category file names cannot be empty.")
                file_name = (
                    file_token
                    if file_token.lower().endswith(".json")
                    else f"{file_token}.json"
                )
                display_name = _derive_name(file_name)
            elif isinstance(entry, dict):
                display_name = str(entry.get("name") or "").strip()
                file_token = str(entry.get("file") or "").strip()
                if not file_token:
                    raise ValueError("Category entries require a 'file' value.")
                file_name = file_token
                if not display_name:
                    display_name = _derive_name(file_name)
            else:
                raise ValueError("Catalog categories must be strings or objects.")

            if not file_name.lower().endswith(".json"):
                file_name = f"{file_name}.json"

            if not display_name:
                display_name = _derive_name(file_name)

            if display_name in seen_names:
                raise ValueError(f"Duplicate category name detected: {display_name}")
            seen_names.add(display_name)

            file_path = Path(file_name)
            if not file_path.is_absolute():
                file_path = data_dir / file_path

            definitions.append(
                CategoryDefinition(name=display_name, path=file_path, group=group_name)
            )
            group_categories.append(display_name)

        if group_categories:
            grouped_names.append((group_name, group_categories))

    if not definitions:
        raise ValueError("The catalog did not define any categories.")

    return definitions, grouped_names


class AudioManager:
    """Generate and play cached speech files for Korean vocabulary words."""

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.voices: Tuple[str, ...] = (
            "ko-KR-SunHiNeural",
            "ko-KR-InJoonNeural",
        )
        self._mutex = threading.Lock()
        self._playback_index: Dict[str, int] = {}

    def play(self, korean: str) -> None:
        """Start background playback of the next cached voice for the word."""
        threading.Thread(
            target=self._prepare_and_play, args=(korean,), daemon=True
        ).start()

    def ensure_audio_files(self, korean: str) -> List[Path]:
        """Return cached audio file paths; try to generate any missing.

        If generation fails (e.g., offline), returns any files that already
        exist so playback still works from cache.
        """
        with self._mutex:
            slug = self._slugify(korean)
            paths = [
                self.base_dir / f"{slug}_{index}.mp3"
                for index in range(len(self.voices))
            ]
            for voice, path in zip(self.voices, paths):
                if not path.exists():
                    try:
                        self._synthesise(korean, voice, path)
                    except Exception as error:  # noqa: BLE001
                        print(f"TTS generation failed for '{korean}' ({voice}): {error}")
                        # Try refreshing known voices once
                        try:
                            asyncio.run(self._discover_ko_voices())
                        except Exception:
                            pass
            available = [p for p in paths if p.exists()]
            if not available:
                raise RuntimeError("No audio available and generation failed.")
            return available

    def _prepare_and_play(self, korean: str) -> None:
        try:
            paths = self.ensure_audio_files(korean)
        except Exception as error:  # noqa: BLE001
            print(f"Audio generation failed for '{korean}': {error}")
            return

        path = self._select_next_path(korean, paths)
        self._play_file(path)

    def _select_next_path(self, korean: str, paths: List[Path]) -> Path:
        with self._mutex:
            index = self._playback_index.get(korean, -1)
            index = (index + 1) % len(paths)
            self._playback_index[korean] = index
            return paths[index]

    def _play_file(self, path: Path) -> None:
        system = platform.system()
        if system == "Windows":
            uri = path.resolve().as_uri()
            script = (
                "Add-Type -AssemblyName PresentationCore;"
                "$player = New-Object System.Windows.Media.MediaPlayer;"
                f"$player.Open([uri]'{uri}');"
                "$player.Play();"
                "while(-not $player.NaturalDuration.HasTimeSpan){Start-Sleep -Milliseconds 50};"
                "while($player.Position -lt $player.NaturalDuration.TimeSpan){Start-Sleep -Milliseconds 100}"
            )
            try:
                subprocess.Popen(
                    [
                        "powershell.exe",
                        "-NoProfile",
                        "-WindowStyle",
                        "Hidden",
                        "-Command",
                        script,
                    ]
                )
            except OSError as error:
                print(f"Unable to launch audio playback for '{path}': {error}")
        else:
            player = None
            for candidate in ("afplay", "mpg123", "mpg321", "play", "aplay", "mpv"):
                resolved = shutil.which(candidate)
                if resolved:
                    player = resolved
                    break
            if player:
                try:
                    subprocess.Popen([player, str(path)])
                except OSError as error:
                    print(f"Unable to launch audio player for '{path}': {error}")
            else:
                print(f"No supported audio player found to play '{path}'.")

    def _synthesise(self, text: str, voice: str, target: Path) -> None:
        async def _runner() -> None:
            communicate = edge_tts.Communicate(text=text, voice=voice)
            await communicate.save(str(target))

        target.parent.mkdir(parents=True, exist_ok=True)
        asyncio.run(_runner())

    @staticmethod
    def _slugify(text: str) -> str:
        safe = "".join(ch if ch.isalnum() else "_" for ch in text).strip("_")
        digest = hashlib.sha1(text.encode("utf-8")).hexdigest()[:10]
        if safe:
            safe = safe[:40].lower()
            return f"{safe}_{digest}"
        return digest

    async def _discover_ko_voices(self) -> None:
        try:
            vm = await edge_tts.VoicesManager.create()
            kos = [v['ShortName'] for v in vm.voices if v.get('Locale') == 'ko-KR']
            # Prefer a female+male pairing if available
            pref: List[str] = []
            for name in ('ko-KR-SunHiNeural','ko-KR-YoungmiNeural','ko-KR-YuJinNeural'):
                if name in kos:
                    pref.append(name)
            for name in ('ko-KR-InJoonNeural','ko-KR-HyunsuNeural','ko-KR-BongJinNeural'):
                if name in kos and name not in pref:
                    pref.append(name)
            # Fill with any remaining ko voices
            for name in kos:
                if name not in pref:
                    pref.append(name)
            if len(pref) >= 2:
                self.voices = (pref[0], pref[1])
        except Exception:
            pass


class QuestionCard:
    """UI component that displays a single category question and answers.

    Each card has a fixed mode (fresh/review) but can rotate categories
    randomly after each answer.
    """

    def __init__(
        self,
        parent: tk.Widget,
        category: str,
        mode: str,
        get_question: Callable[[str, str], Question],
        on_answer: Callable[[str, bool, str], None],
        play_audio: Optional[Callable[[str], None]] = None,
        all_categories: Optional[List[str]] = None,
        answer_mode: str = "choice",
        get_answers: Optional[Callable[[str], List[str]]] = None,
    ) -> None:
        self.category = category
        self.mode = mode
        self.answer_mode = answer_mode
        self.get_question = get_question
        self.on_answer = on_answer
        self.current_question: Optional[Question] = None
        self.active = True
        self.play_audio = play_audio
        self.all_categories = list(all_categories or [category])
        self.get_answers = get_answers or (lambda _category: [])

        self.frame = tk.LabelFrame(
            parent,
            text=self._title_text(),
            padx=16,
            pady=16,
        )
        self.question_var = tk.StringVar()
        self.feedback_var = tk.StringVar()

        self.question_label = tk.Label(
            self.frame,
            textvariable=self.question_var,
            wraplength=320,
            justify="left",
            font=("Segoe UI", 20, "bold"),
        )
        self.question_label.pack(anchor="w", pady=(0, 16))

        self.answer_container = tk.Frame(self.frame)
        self.answer_container.pack(fill="x")

        self.choice_frame = tk.Frame(self.answer_container)
        self.buttons: List[tk.Button] = []
        self._default_button_styles: List[Dict[str, str]] = []
        for index in range(4):
            button = tk.Button(
                self.choice_frame,
                text="",
                font=("Segoe UI", 16),
                command=lambda idx=index: self.submit(idx),
            )
            button.pack(fill="x", pady=6, ipady=8)
            self.buttons.append(button)
            self._default_button_styles.append(
                {
                    "background": button.cget("background"),
                    "foreground": button.cget("foreground"),
                    "activebackground": button.cget("activebackground"),
                    "activeforeground": button.cget("activeforeground"),
                }
            )

        self.type_frame = tk.Frame(self.answer_container)
        entry_row = tk.Frame(self.type_frame)
        entry_row.pack(fill="x")
        self.answer_var = tk.StringVar()
        self.answer_entry = tk.Entry(
            entry_row,
            textvariable=self.answer_var,
            font=("Segoe UI", 16),
        )
        self.answer_entry.pack(side="left", fill="x", expand=True)
        self.answer_entry.bind("<Return>", self.submit_text)
        self.answer_entry.bind("<KeyRelease>", self._handle_entry_changed)
        self.answer_entry.bind("<FocusIn>", self._handle_entry_changed)
        self._entry_default_bg = self.answer_entry.cget("background")

        self.type_submit_button = tk.Button(
            entry_row,
            text="Submit",
            font=("Segoe UI", 12),
            command=self.submit_text,
        )
        self.type_submit_button.pack(side="left", padx=(8, 0))

        helpers_row = tk.Frame(self.type_frame)
        helpers_row.pack(anchor="w", pady=(6, 0), fill="x")
        self.type_help_var = tk.BooleanVar(value=False)
        self.type_help_check = tk.Checkbutton(
            helpers_row,
            text="Type help",
            variable=self.type_help_var,
            command=self._handle_entry_changed,
        )
        self.type_help_check.pack(side="left")

        self.suggestion_box = tk.Listbox(
            self.type_frame,
            height=5,
            activestyle="dotbox",
            exportselection=False,
        )
        self.suggestion_box.bind("<<ListboxSelect>>", self._apply_suggestion)
        self.suggestions_visible = False

        self.feedback_label = tk.Label(
            self.frame,
            textvariable=self.feedback_var,
            fg="#555555",
            wraplength=320,
            justify="left",
            font=("Segoe UI", 12),
        )
        self.feedback_label.pack(anchor="w", pady=(10, 0))

        self._configure_answer_mode()
        self.load_question()

    def _title_text(self) -> str:
        mode_label = "New Focus" if self.mode == "fresh" else "Tough Review"
        return f"{self.category}: {mode_label}"

    def _rotate_category(self) -> None:
        if not self.all_categories:
            return
        choices = [c for c in self.all_categories if c != self.category]
        if choices:
            self.category = random.choice(choices)

    def set_categories(self, categories: List[str]) -> None:
        """Update the allowed categories for this card and refresh its question."""
        if not categories:
            return
        self.all_categories = list(categories)
        if self.category not in self.all_categories:
            self.category = random.choice(self.all_categories)
        self.load_question()
        if self.answer_mode == "type":
            self._handle_entry_changed()

    def _apply_button_style(self, button: tk.Button, bg: str, fg: str) -> None:
        button.config(
            background=bg,
            foreground=fg,
            activebackground=bg,
            activeforeground=fg,
        )

    def _configure_answer_mode(self) -> None:
        if self.answer_mode == "type":
            self.choice_frame.pack_forget()
            self.type_frame.pack(fill="x")
        else:
            self.type_frame.pack_forget()
            self.choice_frame.pack(fill="x")
            self._hide_suggestions()

    def load_question(self) -> None:
        """Populate the card with a fresh question and answer options."""
        # Update frame title, in case category was rotated.
        self.frame.config(text=self._title_text())
        try:
            question = self.get_question(self.category, self.mode)
        except ValueError as error:
            self.question_var.set("Unable to generate question.")
            self.feedback_var.set(str(error))
            for button in self.buttons:
                button.config(text="", state=tk.DISABLED)
            if self.answer_mode == "type":
                self.answer_entry.config(state=tk.DISABLED)
                self.type_submit_button.config(state=tk.DISABLED)
                self._hide_suggestions()
            self.active = False
            return

        self.current_question = question
        self.question_var.set(question.prompt)
        self.feedback_var.set("")
        self.active = True
        if self.answer_mode == "type":
            self.answer_var.set("")
            self.answer_entry.config(state=tk.NORMAL, background=self._entry_default_bg)
            self.type_submit_button.config(state=tk.NORMAL)
            self._hide_suggestions()

        for index, (button, option) in enumerate(zip(self.buttons, question.options)):
            defaults = self._default_button_styles[index]
            button.config(
                text=option,
                state=tk.NORMAL,
                background=defaults["background"],
                foreground=defaults["foreground"],
                activebackground=defaults["activebackground"],
                activeforeground=defaults["activeforeground"],
            )
            button.config(state=tk.NORMAL)

    def set_answer_mode(self, mode: str) -> None:
        """Switch between multiple choice and type mode."""
        if mode not in {"choice", "type"}:
            return
        if mode == self.answer_mode:
            return
        self.answer_mode = mode
        self._configure_answer_mode()
        self.load_question()

    def _handle_entry_changed(self, _event: Optional[tk.Event] = None) -> None:
        if self.answer_mode != "type":
            return
        if not self.type_help_var.get():
            self._hide_suggestions()
            return
        query = self.answer_var.get().strip()
        if not query:
            self._hide_suggestions()
            return
        lowered_query = query.lower()
        matches: List[str] = []
        for answer in self.get_answers(self.category):
            if lowered_query in answer.lower():
                matches.append(answer)
                if len(matches) >= 8:
                    break
        if not matches:
            self._hide_suggestions()
            return
        self.suggestion_box.delete(0, tk.END)
        for match in matches:
            self.suggestion_box.insert(tk.END, match)
        if not self.suggestions_visible:
            self.suggestion_box.pack(fill="x", pady=(6, 0))
            self.suggestions_visible = True

    def _apply_suggestion(self, _event: Optional[tk.Event] = None) -> None:
        selection = self.suggestion_box.curselection()
        if not selection:
            return
        value = self.suggestion_box.get(selection[0])
        self.answer_var.set(value)
        self.answer_entry.icursor(tk.END)
        self._hide_suggestions()

    def _hide_suggestions(self) -> None:
        if self.suggestions_visible:
            self.suggestion_box.pack_forget()
            self.suggestions_visible = False
        self.suggestion_box.selection_clear(0, tk.END)

    def _handle_result(self, is_correct: bool) -> None:
        if self.current_question is None:
            return
        self.on_answer(self.category, is_correct, self.current_question.korean)
        if self.play_audio:
            self.play_audio(self.current_question.korean)

    def _queue_next_question(self) -> None:
        self._rotate_category()
        self.frame.after(1500, self.load_question)

    def submit(self, index: int) -> None:
        """Handle a user selecting one of the answer buttons."""
        if self.answer_mode != "choice":
            return
        if not self.active or self.current_question is None:
            return
        if index >= len(self.current_question.options):
            return

        choice = self.current_question.options[index]
        is_correct = choice == self.current_question.answer
        self._handle_result(is_correct)

        selected_button = self.buttons[index]
        if is_correct:
            self.feedback_var.set("Correct!")
            self._apply_button_style(selected_button, "#2ecc71", "#ffffff")
        else:
            self.feedback_var.set(f"Correct answer: {self.current_question.answer}")
            self._apply_button_style(selected_button, "#e74c3c", "#ffffff")
            try:
                correct_index = self.current_question.options.index(
                    self.current_question.answer
                )
            except ValueError:
                correct_index = None
            if correct_index is not None and 0 <= correct_index < len(self.buttons):
                correct_button = self.buttons[correct_index]
                self._apply_button_style(correct_button, "#2ecc71", "#ffffff")

        self.active = False
        for button in self.buttons:
            button.config(state=tk.DISABLED)

        # Rotate to a random other category for this panel, then queue next question.
        self._queue_next_question()

    def submit_text(self, _event: Optional[tk.Event] = None) -> None:
        """Evaluate a typed answer."""
        if self.answer_mode != "type":
            return
        if not self.active or self.current_question is None:
            return
        guess = self.answer_var.get()
        if not guess.strip():
            self.feedback_var.set("Please type an answer before submitting.")
            return
        is_correct = normalize_answer(guess) == normalize_answer(
            self.current_question.answer
        )
        self._handle_result(is_correct)
        if is_correct:
            self.feedback_var.set("Correct!")
            self.answer_entry.config(background="#d4edda")
        else:
            self.feedback_var.set(f"Correct answer: {self.current_question.answer}")
            self.answer_entry.config(background="#f8d7da")
        self.answer_entry.config(state=tk.DISABLED)
        self.type_submit_button.config(state=tk.DISABLED)
        self.active = False
        self._hide_suggestions()
        self._queue_next_question()

    def set_font_scheme(self, scheme: Dict[str, Tuple[Any, ...]]) -> None:
        """Apply the supplied font scheme to labels and buttons."""
        question_font = scheme.get("question", ("Segoe UI", 20, "bold"))
        option_font = scheme.get("option", ("Segoe UI", 16))
        feedback_font = scheme.get("feedback", ("Segoe UI", 12))
        self.question_label.config(font=question_font)
        for button in self.buttons:
            button.config(font=option_font)
        self.answer_entry.config(font=option_font)
        self.type_submit_button.config(font=option_font)
        self.feedback_label.config(font=feedback_font)


class QuizApp:
    """Main application window that coordinates question cards and scoring."""

    def __init__(
        self,
        root: tk.Tk,
        vocabulary: Dict[str, Vocabulary],
        progress: Progress,
        progress_path: Path,
        audio_manager: AudioManager,
        metadata: Dict[str, Any],
        category_groups: Optional[List[Tuple[str, List[str]]]] = None,
    ) -> None:
        self.root = root
        self.vocabulary = vocabulary
        self.progress = progress
        self.progress_path = progress_path
        self.audio_manager = audio_manager
        self.save_error_reported = False
        self.categories = list(vocabulary.keys())
        self.active_categories = list(self.categories)
        self.cards: List[QuestionCard] = []
        self.translation_direction = "ko-en"
        self.answer_mode = "choice"
        if category_groups:
            self.category_groups = category_groups
        else:
            self.category_groups = [("All sections", self.categories)]

        self.current_streak = int(metadata.get("streak", 0) or 0)
        self.longest_streak = int(metadata.get("longest_streak", 0) or 0)
        self.streak_timestamp = str(metadata.get("streak_timestamp") or "")
        self.longest_streak_timestamp = str(
            metadata.get("longest_streak_timestamp") or ""
        )

        self.root.title("Korean Vocabulary Quiz")
        self.root.configure(padx=18, pady=18)
        self.root.minsize(960, 520)

        header_frame = tk.Frame(root)
        header_frame.pack(fill="x")

        info_frame = tk.Frame(header_frame)
        info_frame.pack(side="left", fill="x", expand=True)

        title_label = tk.Label(
            info_frame, text="Korean Vocabulary Quiz", font=("Segoe UI", 22, "bold")
        )
        title_label.pack(anchor="w")

        subtitle = tk.Label(
            info_frame,
            text=(
                "Top row focuses on new or rarely seen words. Bottom row revisits the toughest words "
                "so far. New questions appear immediately and progress is saved as you go."
            ),
            wraplength=720,
            justify="left",
        )
        subtitle.pack(anchor="w", pady=(6, 0))

        controls_frame = tk.Frame(header_frame)
        controls_frame.pack(side="left", padx=(12, 0))
        self.section_button = tk.Button(
            controls_frame,
            text="Choose Sections…",
            font=("Segoe UI", 11),
            command=self._toggle_selection_panel,
        )
        self.section_button.pack(anchor="w")
        self.section_status_var = tk.StringVar()
        self._update_section_status()
        section_status_label = tk.Label(
            controls_frame,
            textvariable=self.section_status_var,
            font=("Segoe UI", 9),
            fg="#444444",
            justify="left",
            wraplength=200,
        )
        section_status_label.pack(anchor="w", pady=(4, 0))
        self.answer_mode_var = tk.StringVar(value=self.answer_mode)
        answer_mode_frame = tk.Frame(controls_frame)
        answer_mode_frame.pack(anchor="w", pady=(10, 0), fill="x")
        answer_mode_label = tk.Label(
            answer_mode_frame, text="Answer mode:", font=("Segoe UI", 10, "bold")
        )
        answer_mode_label.pack(anchor="w")
        answer_mode_row = tk.Frame(answer_mode_frame)
        answer_mode_row.pack(anchor="w", pady=(4, 0))
        answer_mode_options = (("choice", "Multiple choice"), ("type", "Type mode"))
        for index, (value, label) in enumerate(answer_mode_options):
            button = tk.Radiobutton(
                answer_mode_row,
                text=label,
                value=value,
                variable=self.answer_mode_var,
                indicatoron=False,
                width=14,
                command=self._handle_answer_mode_change,
            )
            button.pack(side="left", padx=(0 if index == 0 else 6, 0))
        self.score_toggle_button = tk.Button(
            controls_frame,
            text="Hide Scores",
            font=("Segoe UI", 10),
            command=self._toggle_scoreboard_visibility,
        )
        self.score_toggle_button.pack(anchor="w", pady=(8, 0))
        self.scoreboard_visible = True
        self.font_presets: Dict[str, Dict[str, Tuple[Any, ...]]] = {
            "small": {
                "question": ("Segoe UI", 16, "bold"),
                "option": ("Segoe UI", 12),
                "feedback": ("Segoe UI", 10),
            },
            "normal": {
                "question": ("Segoe UI", 18, "bold"),
                "option": ("Segoe UI", 14),
                "feedback": ("Segoe UI", 11),
            },
            "large": {
                "question": ("Segoe UI", 20, "bold"),
                "option": ("Segoe UI", 16),
                "feedback": ("Segoe UI", 12),
            },
        }
        self.font_size_var = tk.StringVar(value="large")
        self.current_font_size = self.font_size_var.get()
        font_controls = tk.Frame(controls_frame)
        font_controls.pack(anchor="w", pady=(10, 0), fill="x")
        font_label = tk.Label(font_controls, text="Font size:", font=("Segoe UI", 10, "bold"))
        font_label.pack(anchor="w")
        buttons_row = tk.Frame(font_controls)
        buttons_row.pack(anchor="w", pady=(4, 0))
        self.font_size_buttons: Dict[str, tk.Radiobutton] = {}
        for index, (size_key, label_text) in enumerate(
            (("small", "Small"), ("normal", "Normal"), ("large", "Large"))
        ):
            button = tk.Radiobutton(
                buttons_row,
                text=label_text,
                value=size_key,
                variable=self.font_size_var,
                indicatoron=False,
                width=8,
                command=self._handle_font_size_button,
            )
            button.pack(side="left", padx=(0 if index == 0 else 6, 0))
            self.font_size_buttons[size_key] = button
        self.direction_var = tk.StringVar(value=self.translation_direction)
        direction_controls = tk.Frame(controls_frame)
        direction_controls.pack(anchor="w", pady=(10, 0), fill="x")
        direction_label = tk.Label(
            direction_controls, text="Direction:", font=("Segoe UI", 10, "bold")
        )
        direction_label.pack(anchor="w")
        direction_row = tk.Frame(direction_controls)
        direction_row.pack(anchor="w", pady=(4, 0))
        direction_options = (
            ("ko-en", "Korean -> English"),
            ("en-ko", "English -> Korean"),
        )
        for index, (dir_key, dir_label) in enumerate(direction_options):
            button = tk.Radiobutton(
                direction_row,
                text=dir_label,
                value=dir_key,
                variable=self.direction_var,
                indicatoron=False,
                width=18,
                command=self._handle_direction_change,
            )
            button.pack(side="left", padx=(0 if index == 0 else 6, 0))

        self.scoreboard_frame = tk.Frame(header_frame)
        self.scoreboard_frame.pack(side="right", anchor="ne", padx=(18, 0))
        for col in range(2):
            self.scoreboard_frame.grid_columnconfigure(col, weight=1)

        score_title = tk.Label(
            self.scoreboard_frame, text="Scores", font=("Segoe UI", 11, "bold")
        )
        score_title.grid(row=0, column=0, columnspan=2, sticky="e", pady=(0, 4))

        self.streak_var = tk.StringVar()
        self.longest_streak_var = tk.StringVar()
        streak_label = tk.Label(
            self.scoreboard_frame,
            textvariable=self.streak_var,
            font=("Segoe UI", 10),
            anchor="e",
            justify="right",
        )
        streak_label.grid(row=1, column=0, columnspan=2, sticky="e")

        longest_label = tk.Label(
            self.scoreboard_frame,
            textvariable=self.longest_streak_var,
            font=("Segoe UI", 10),
            anchor="e",
            justify="right",
        )
        longest_label.grid(row=2, column=0, columnspan=2, sticky="e", pady=(0, 4))

        self.score_labels: Dict[str, tk.Label] = {}
        for index, category in enumerate(self.categories):
            row = index // 2 + 3
            column = index % 2
            label = tk.Label(
                self.scoreboard_frame,
                text="",
                font=("Segoe UI", 10),
                anchor="e",
                justify="right",
            )
            label.grid(row=row, column=column, sticky="e", padx=(12 if column else 0, 0))
            self.score_labels[category] = label

        self.selection_panel = tk.Frame(root, bd=1, relief="groove", padx=12, pady=12)
        self.selection_panel_visible = False
        panel_title = tk.Label(
            self.selection_panel, text="Select sections to study", font=("Segoe UI", 12, "bold")
        )
        panel_title.pack(anchor="w")
        panel_hint = tk.Label(
            self.selection_panel,
            text=(
                "Pick one or more sections to focus on. Sections are grouped according to "
                "the catalog.json file so lessons and vocab sets stay organised."
            ),
            justify="left",
            wraplength=720,
        )
        panel_hint.pack(anchor="w", pady=(4, 8))
        options_frame = tk.Frame(self.selection_panel)
        options_frame.pack(fill="x")
        self.category_vars: Dict[str, tk.BooleanVar] = {}
        groups_to_render = self.category_groups or [("All sections", self.categories)]
        for index, (group_name, group_categories) in enumerate(groups_to_render):
            group_frame = tk.LabelFrame(options_frame, text=group_name)
            group_frame.grid(row=0, column=index, sticky="nsew", padx=6, pady=6)
            options_frame.grid_columnconfigure(index, weight=1)
            for category in group_categories:
                if category not in self.category_vars:
                    self.category_vars[category] = tk.BooleanVar(value=True)
                checkbox = tk.Checkbutton(
                    group_frame,
                    text=category,
                    variable=self.category_vars[category],
                    anchor="w",
                    justify="left",
                    wraplength=220,
                )
                checkbox.pack(fill="x", anchor="w", padx=6, pady=2)
        panel_buttons = tk.Frame(self.selection_panel)
        panel_buttons.pack(fill="x", pady=(8, 0))
        select_all_button = tk.Button(
            panel_buttons, text="Select All", command=self._select_all_sections
        )
        select_all_button.pack(side="left")
        clear_all_button = tk.Button(
            panel_buttons, text="Clear All", command=self._clear_all_sections
        )
        clear_all_button.pack(side="left", padx=(8, 0))
        close_button = tk.Button(
            panel_buttons, text="Close", command=lambda: self._toggle_selection_panel(False)
        )
        close_button.pack(side="right")
        apply_button = tk.Button(
            panel_buttons, text="Apply Selection", command=self._apply_section_selection
        )
        apply_button.pack(side="right", padx=(0, 8))

        self.cards_frame = tk.Frame(root)
        self.cards_frame.pack(fill="both", expand=True)

        categories = self.active_categories

        # Build 3 columns x 2 rows: top fresh, bottom review.
        num_cols = 3
        for c in range(num_cols):
            self.cards_frame.columnconfigure(c, weight=1)
        for row, mode in enumerate(("fresh", "review")):
            self.cards_frame.rowconfigure(row, weight=1)
            for col in range(num_cols):
                initial_category = random.choice(categories)
                card = QuestionCard(
                    self.cards_frame,
                    initial_category,
                    mode,
                    self.generate_question,
                    self.handle_answer,
                    self.audio_manager.play,
                    all_categories=categories,
                    answer_mode=self.answer_mode,
                    get_answers=self.get_category_answers,
                )
                card.frame.grid(row=row, column=col, padx=8, pady=6, sticky="nsew")
                card.set_font_scheme(self.font_presets[self.current_font_size])
                self.cards.append(card)

        self.update_scoreboard()

    def generate_question(self, category: str, mode: str) -> Question:
        """Return the next weighted question for a category."""
        pool = self.vocabulary[category]
        stats = self.progress.setdefault(category, {})
        return build_question(pool, stats, mode, self.translation_direction)

    def get_category_answers(self, category: str) -> List[str]:
        """Return all possible answers for the current direction."""
        pool = self.vocabulary.get(category, [])
        if self.translation_direction == "ko-en":
            return [english for _, english in pool]
        return [korean for korean, _ in pool]

    def handle_answer(self, category: str, is_correct: bool, korean_word: str) -> None:
        """Update scores and refresh the scoreboard when a card reports an answer."""
        stats = self.progress.setdefault(category, {})
        record = stats.setdefault(korean_word, {"history": []})
        history = record.setdefault("history", [])
        history.append(1 if is_correct else 0)
        if len(history) > 10:
            del history[:-10]

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        if is_correct:
            self.current_streak += 1
            self.streak_timestamp = timestamp
            if self.current_streak > self.longest_streak:
                self.longest_streak = self.current_streak
                self.longest_streak_timestamp = timestamp
        else:
            self.current_streak = 0
            self.streak_timestamp = timestamp

        self.update_scoreboard()
        self.save_progress()

    def update_scoreboard(self) -> None:
        """Display the latest per-category scores."""
        for category in self.categories:
            stats = self.progress.get(category, {})
            attempts = 0
            correct = 0
            for record in stats.values():
                history = record.get("history", [])
                attempts += len(history)
                correct += sum(history)
            if attempts:
                accuracy = (correct / attempts) * 100
                text = f"{category}: {correct}/{attempts} ({accuracy:.0f}%)"
            else:
                text = f"{category}: 0/0"
            label = self.score_labels.get(category)
            if label:
                label.config(text=text)
        self._update_streak_labels()

    def _update_streak_labels(self) -> None:
        current_stamp = self._format_timestamp(self.streak_timestamp)
        longest_stamp = self._format_timestamp(self.longest_streak_timestamp)
        self.streak_var.set(f"Streak: {self.current_streak} ({current_stamp})")
        self.longest_streak_var.set(
            f"Longest: {self.longest_streak} ({longest_stamp})"
        )

    @staticmethod
    def _format_timestamp(stamp: str) -> str:
        return stamp if stamp else "n/a"

    def _toggle_selection_panel(self, show: Optional[bool] = None) -> None:
        """Show or hide the section selection panel."""
        if show is None:
            show = not self.selection_panel_visible
        if show:
            self._sync_section_checkboxes()
            if not self.selection_panel_visible:
                self.selection_panel.pack(
                    before=self.cards_frame, fill="x", pady=(12, 12)
                )
        else:
            if self.selection_panel_visible:
                self.selection_panel.pack_forget()
        self.selection_panel_visible = show

    def _toggle_scoreboard_visibility(self) -> None:
        """Hide or show the per-section score list."""
        if self.scoreboard_visible:
            self.scoreboard_frame.pack_forget()
            self.score_toggle_button.config(text="Show Scores")
        else:
            self.scoreboard_frame.pack(side="right", anchor="ne", padx=(18, 0))
            self.score_toggle_button.config(text="Hide Scores")
        self.scoreboard_visible = not self.scoreboard_visible

    def _handle_font_size_button(self) -> None:
        """Respond to font size button presses."""
        self._set_font_size(self.font_size_var.get())

    def _set_font_size(self, size: str) -> None:
        """Apply the requested font preset to all cards."""
        if size not in self.font_presets:
            size = "large"
            self.font_size_var.set(size)
        scheme = self.font_presets[size]
        self.current_font_size = size
        for card in self.cards:
            card.set_font_scheme(scheme)

    def _handle_answer_mode_change(self) -> None:
        """Switch between multiple-choice and typing practice."""
        mode = self.answer_mode_var.get()
        if mode not in ("choice", "type"):
            mode = "choice"
            self.answer_mode_var.set(mode)
        if mode == self.answer_mode:
            return
        self.answer_mode = mode
        for card in self.cards:
            card.set_answer_mode(mode)

    def _handle_direction_change(self) -> None:
        """Switch between Korean→English and English→Korean practice."""
        new_direction = self.direction_var.get()
        if new_direction not in ("ko-en", "en-ko"):
            new_direction = "ko-en"
            self.direction_var.set(new_direction)
        if new_direction == self.translation_direction:
            return
        self.translation_direction = new_direction
        for card in self.cards:
            card.load_question()

    def _sync_section_checkboxes(self) -> None:
        active = set(self.active_categories)
        for category, var in self.category_vars.items():
            var.set(category in active)

    def _select_all_sections(self) -> None:
        for var in self.category_vars.values():
            var.set(True)

    def _clear_all_sections(self) -> None:
        for var in self.category_vars.values():
            var.set(False)

    def _apply_section_selection(self) -> None:
        selection = [
            category for category, var in self.category_vars.items() if var.get()
        ]
        if not selection:
            messagebox.showwarning(
                "No sections selected", "Please select at least one section to study."
            )
            return
        self.active_categories = selection
        for card in self.cards:
            card.set_categories(self.active_categories)
        self._update_section_status()
        self._toggle_selection_panel(False)

    def _update_section_status(self) -> None:
        total = len(self.categories)
        count = len(self.active_categories)
        if count == total:
            summary = "Training: All sections"
        elif count == 1:
            summary = f"Training: {self.active_categories[0]}"
        else:
            summary = f"Training: {count} of {total} sections"
        self.section_status_var.set(summary)

    def save_progress(self) -> None:
        """Persist the progress file to disk."""
        payload: Dict[str, Any] = {"__meta__": self._serialize_metadata()}
        for category in self.categories:
            stats = self.progress.get(category, {})
            category_payload: CategoryStats = {}
            for korean, record in stats.items():
                history = [1 if bool(value) else 0 for value in record.get("history", [])]
                history = history[-10:]
                if history:
                    category_payload[korean] = {"history": history}
            payload[category] = category_payload

        try:
            with self.progress_path.open("w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
        except OSError as error:
            if not self.save_error_reported:
                messagebox.showwarning("Save failed", f"Could not save progress:\n{error}")
                self.save_error_reported = True
        else:
            self.save_error_reported = False

    def _serialize_metadata(self) -> Dict[str, Any]:
        return {
            "streak": self.current_streak,
            "streak_timestamp": self.streak_timestamp,
            "longest_streak": self.longest_streak,
            "longest_streak_timestamp": self.longest_streak_timestamp,
        }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch a windowed Korean vocabulary quiz with three concurrent questions."
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directory containing JSON vocab/lesson files plus catalog.json.",
    )
    parser.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help=(
            "Optional path to a catalog JSON file that defines groups of lesson and vocabulary datasets."
        ),
    )
    args = parser.parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)
    catalog_path = args.catalog if args.catalog else args.data_dir / "catalog.json"

    root = tk.Tk()
    root.withdraw()

    progress_path = args.data_dir / "progress.json"
    audio_dir = args.data_dir / "audio"

    try:
        category_definitions, grouped_categories = load_category_catalog(
            args.data_dir, catalog_path
        )
        vocabulary = {
            definition.name: load_vocab(definition.path)
            for definition in category_definitions
        }
        progress, metadata = load_progress(
            progress_path, [definition.name for definition in category_definitions]
        )
    except (FileNotFoundError, json.JSONDecodeError, KeyError, ValueError, OSError) as error:
        messagebox.showerror("Unable to load data", str(error))
        root.destroy()
        return

    root.deiconify()
    audio_manager = AudioManager(audio_dir)
    QuizApp(
        root,
        vocabulary,
        progress,
        progress_path,
        audio_manager,
        metadata,
        grouped_categories,
    )
    root.mainloop()


if __name__ == "__main__":
    main()





