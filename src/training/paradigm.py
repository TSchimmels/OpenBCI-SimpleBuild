"""Graz motor imagery calibration paradigm.

Implements the visual cueing protocol described in:

    Pfurtscheller, G. & Neuper, C. (2001). Motor imagery and direct
    brain-computer communication. Proceedings of the IEEE, 89(7), 1123-1134.

The paradigm displays visual cues (arrows) on screen and records event
markers into a DataRecorder so that EEG epochs can later be extracted and
labelled for offline classifier training.
"""

from __future__ import annotations

import logging
import random
import time
from typing import TYPE_CHECKING, Dict, List, Optional

import numpy as np
import pygame

if TYPE_CHECKING:
    from .recorder import DataRecorder

logger = logging.getLogger(__name__)


class GrazParadigm:
    """Graz motor imagery calibration paradigm.

    Presents a sequence of trials, each consisting of:

    1. **Fixation cross** -- white ``+`` on a black background (default 2 s).
    2. **Beep + cue** -- short audio tone and a directional arrow (or no
       arrow for *rest*) indicating the motor imagery task.
    3. **Imagery period** -- the cue remains visible while the participant
       performs motor imagery (default 4 s).
    4. **Rest** -- blank screen for a randomised interval (default 1.5--3 s).

    Trials are pseudo-randomised in blocks so that each class appears an
    equal number of times per block.  Event markers are sent to a
    :class:`DataRecorder` at cue onset.

    Args:
        config: Full application configuration dictionary (as loaded from
            ``config/settings.yaml``).  The ``training`` section is used.
    """

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    def __init__(self, config: Dict) -> None:
        train_cfg: Dict = config.get("training", {})

        self.n_classes: int = train_cfg.get("n_classes", 2)
        self.classes: List[str] = train_cfg.get("classes", ["left_hand", "right_hand"])
        self.n_trials_per_class: int = train_cfg.get("n_trials_per_class", 40)
        self.n_runs: int = train_cfg.get("n_runs", 1)

        # Timing (seconds)
        self.fixation_duration: float = train_cfg.get("fixation_duration", 2.0)
        self.cue_duration: float = train_cfg.get("cue_duration", 1.25)
        self.imagery_duration: float = train_cfg.get("imagery_duration", 4.0)
        self.rest_duration_min: float = train_cfg.get("rest_duration_min", 1.5)
        self.rest_duration_max: float = train_cfg.get("rest_duration_max", 3.0)

        # Audio
        self.beep_frequency: int = train_cfg.get("beep_frequency", 1000)
        self.beep_duration_ms: int = train_cfg.get("beep_duration_ms", 70)

        # Derived
        self._total_trials: int = self.n_trials_per_class * self.n_classes
        self._trials_per_run: int = self._total_trials // max(self.n_runs, 1)

        # Debug mode: set to True to use a windowed display instead of
        # fullscreen.  Controlled externally by callers if needed.
        self.debug: bool = False

        logger.info(
            "GrazParadigm: %d classes %s, %d trials/class, %d run(s), "
            "%d total trials.",
            self.n_classes,
            self.classes,
            self.n_trials_per_class,
            self.n_runs,
            self._total_trials,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, recorder: "DataRecorder") -> None:
        """Execute the full calibration paradigm.

        Opens a pygame display, presents all trials across all runs,
        and records event markers into *recorder*.  The recorder should
        already be started (``recorder.start()``).

        Press **ESC** at any time to abort.  Between runs, a break
        screen is shown until the participant presses **SPACE**.

        Args:
            recorder: A running :class:`DataRecorder` instance that will
                receive event markers at cue onset.
        """
        pygame.init()
        pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)

        if self.debug:
            screen = pygame.display.set_mode((1024, 768))
        else:
            screen = pygame.display.set_mode((0, 0), pygame.FULLSCREEN)

        pygame.display.set_caption("EEG Cursor — Graz Paradigm")
        pygame.mouse.set_visible(False)

        clock = pygame.time.Clock()

        # Build the full trial sequence across all runs
        trial_sequence = self._build_trial_sequence()
        total_trials = len(trial_sequence)

        logger.info("Starting paradigm: %d total trials.", total_trials)

        try:
            trial_global_idx = 0
            for run_idx in range(self.n_runs):
                # Determine which slice of the trial sequence this run covers
                run_start = run_idx * self._trials_per_run
                run_end = run_start + self._trials_per_run
                # Last run picks up any remainder from integer division
                if run_idx == self.n_runs - 1:
                    run_end = total_trials
                run_trials = trial_sequence[run_start:run_end]

                for local_idx, class_name in enumerate(run_trials):
                    trial_global_idx += 1

                    if self._check_abort():
                        logger.info("Paradigm aborted by user at trial %d.", trial_global_idx)
                        return

                    # --- 1. Fixation cross ---
                    self._timed_display(
                        screen,
                        clock,
                        self.fixation_duration,
                        draw_fn=lambda s: self._draw_fixation(s),
                        progress_text=f"Trial {trial_global_idx} / {total_trials}",
                    )

                    if self._check_abort():
                        return

                    # --- 2. Beep + Cue onset ---
                    self._play_beep(self.beep_frequency, self.beep_duration_ms)

                    # Record event marker at cue onset
                    recorder.add_event(class_name)
                    logger.debug(
                        "Trial %d: cue '%s' at %.3f s.",
                        trial_global_idx,
                        class_name,
                        time.time(),
                    )

                    # --- 3. Cue + imagery period ---
                    cue_total = self.cue_duration + self.imagery_duration

                    if class_name == "rest":
                        draw_fn = lambda s: self._draw_fixation(s)
                    elif class_name == "left_hand":
                        draw_fn = lambda s: (
                            self._draw_fixation(s),
                            self._draw_arrow(s, "left"),
                        )
                    elif class_name == "right_hand":
                        draw_fn = lambda s: (
                            self._draw_fixation(s),
                            self._draw_arrow(s, "right"),
                        )
                    elif class_name == "feet":
                        draw_fn = lambda s: (
                            self._draw_fixation(s),
                            self._draw_arrow(s, "down"),
                        )
                    elif class_name == "tongue":
                        draw_fn = lambda s: (
                            self._draw_fixation(s),
                            self._draw_arrow(s, "up"),
                        )
                    else:
                        # Generic fallback: show the class name as text
                        draw_fn = lambda s, cn=class_name: (
                            self._draw_fixation(s),
                            self._draw_text(s, cn),
                        )

                    self._timed_display(
                        screen,
                        clock,
                        cue_total,
                        draw_fn=draw_fn,
                        progress_text=f"Trial {trial_global_idx} / {total_trials}",
                    )

                    if self._check_abort():
                        return

                    # --- 4. Rest (blank screen, random duration) ---
                    rest_dur = random.uniform(
                        self.rest_duration_min, self.rest_duration_max
                    )
                    self._timed_display(
                        screen,
                        clock,
                        rest_dur,
                        draw_fn=lambda s: s.fill((0, 0, 0)),
                    )

                # Break screen between runs (except after the last run)
                if run_idx < self.n_runs - 1:
                    logger.info("Run %d complete. Showing break screen.", run_idx + 1)
                    self._show_break_screen(screen, clock, run_idx + 1, self.n_runs)

        finally:
            pygame.mouse.set_visible(True)
            pygame.quit()
            logger.info("Paradigm finished. pygame shut down.")

    # ------------------------------------------------------------------
    # Drawing helpers
    # ------------------------------------------------------------------

    def _draw_fixation(self, screen: pygame.Surface) -> None:
        """Draw a white fixation cross centred on the screen.

        Args:
            screen: The pygame display surface.
        """
        screen.fill((0, 0, 0))
        w, h = screen.get_size()
        font = pygame.font.Font(None, 120)
        text_surface = font.render("+", True, (255, 255, 255))
        rect = text_surface.get_rect(center=(w // 2, h // 2))
        screen.blit(text_surface, rect)

    def _draw_arrow(self, screen: pygame.Surface, direction: str) -> None:
        """Draw an arrow pointing in the given direction.

        The arrow is drawn as a filled polygon on the existing screen
        contents (call after ``_draw_fixation`` if you want both).

        For horizontal arrows (left/right), the arrow is drawn above
        the fixation cross.  For vertical arrows (up/down), the arrow
        is drawn above or below the fixation cross respectively.

        Args:
            screen: The pygame display surface.
            direction: ``'left'``, ``'right'``, ``'up'``, or ``'down'``.
        """
        w, h = screen.get_size()
        cx, cy = w // 2, h // 2

        # Arrow geometry: shaft + triangle head
        arrow_len = min(w, h) // 5
        shaft_half_h = arrow_len // 8
        head_half_h = arrow_len // 3
        head_len = arrow_len // 3

        colour = (0, 200, 0)  # Green arrow

        if direction == "right":
            # Position the arrow above the fixation cross
            y_offset = -arrow_len

            # Shaft: rectangle from centre-left to centre-right minus head
            shaft_left = cx - arrow_len // 2
            shaft_right = cx + arrow_len // 2 - head_len
            head_tip = cx + arrow_len // 2

            points = [
                # Shaft (top-left, top-right, head-top, tip, head-bottom,
                # bottom-right, bottom-left)
                (shaft_left, cy + y_offset - shaft_half_h),
                (shaft_right, cy + y_offset - shaft_half_h),
                (shaft_right, cy + y_offset - head_half_h),
                (head_tip, cy + y_offset),
                (shaft_right, cy + y_offset + head_half_h),
                (shaft_right, cy + y_offset + shaft_half_h),
                (shaft_left, cy + y_offset + shaft_half_h),
            ]
        elif direction == "left":
            # Position the arrow above the fixation cross
            y_offset = -arrow_len

            shaft_right = cx + arrow_len // 2
            shaft_left = cx - arrow_len // 2 + head_len
            head_tip = cx - arrow_len // 2

            points = [
                (shaft_right, cy + y_offset - shaft_half_h),
                (shaft_left, cy + y_offset - shaft_half_h),
                (shaft_left, cy + y_offset - head_half_h),
                (head_tip, cy + y_offset),
                (shaft_left, cy + y_offset + head_half_h),
                (shaft_left, cy + y_offset + shaft_half_h),
                (shaft_right, cy + y_offset + shaft_half_h),
            ]
        elif direction == "up":
            # Position the arrow above the fixation cross
            x_offset = 0

            # Vertical arrow pointing up: shaft runs vertically,
            # triangle head points upward
            shaft_bottom = cy - arrow_len // 4
            shaft_top = cy - arrow_len + head_len
            head_tip = cy - arrow_len

            points = [
                # Shaft (bottom-left, top-left, head-left, tip,
                # head-right, top-right, bottom-right)
                (cx + x_offset - shaft_half_h, shaft_bottom),
                (cx + x_offset - shaft_half_h, shaft_top),
                (cx + x_offset - head_half_h, shaft_top),
                (cx + x_offset, head_tip),
                (cx + x_offset + head_half_h, shaft_top),
                (cx + x_offset + shaft_half_h, shaft_top),
                (cx + x_offset + shaft_half_h, shaft_bottom),
            ]
        elif direction == "down":
            # Position the arrow below the fixation cross
            x_offset = 0

            # Vertical arrow pointing down: shaft runs vertically,
            # triangle head points downward
            shaft_top = cy + arrow_len // 4
            shaft_bottom = cy + arrow_len - head_len
            head_tip = cy + arrow_len

            points = [
                # Shaft (top-left, bottom-left, head-left, tip,
                # head-right, bottom-right, top-right)
                (cx + x_offset - shaft_half_h, shaft_top),
                (cx + x_offset - shaft_half_h, shaft_bottom),
                (cx + x_offset - head_half_h, shaft_bottom),
                (cx + x_offset, head_tip),
                (cx + x_offset + head_half_h, shaft_bottom),
                (cx + x_offset + shaft_half_h, shaft_bottom),
                (cx + x_offset + shaft_half_h, shaft_top),
            ]
        else:
            logger.warning("Unknown arrow direction '%s'; not drawing.", direction)
            return

        pygame.draw.polygon(screen, colour, points)

    def _draw_text(self, screen: pygame.Surface, text: str) -> None:
        """Draw centred white text on the current screen contents.

        Renders *text* slightly below the centre of the screen so it
        does not overlap with the fixation cross.

        Args:
            screen: The pygame display surface.
            text: The string to render.
        """
        w, h = screen.get_size()
        font = pygame.font.Font(None, 72)
        text_surface = font.render(text, True, (255, 255, 255))
        rect = text_surface.get_rect(center=(w // 2, h // 2 + 100))
        screen.blit(text_surface, rect)

    def _draw_progress(self, screen: pygame.Surface, text: str) -> None:
        """Draw small progress text in the bottom-right corner.

        Args:
            screen: The pygame display surface.
            text: Progress string, e.g. ``"Trial 5 / 40"``.
        """
        w, h = screen.get_size()
        font = pygame.font.Font(None, 32)
        text_surface = font.render(text, True, (100, 100, 100))
        rect = text_surface.get_rect(bottomright=(w - 20, h - 20))
        screen.blit(text_surface, rect)

    # ------------------------------------------------------------------
    # Audio
    # ------------------------------------------------------------------

    def _play_beep(self, freq: int = 1000, duration_ms: int = 70) -> None:
        """Play a short sine-wave beep.

        Args:
            freq: Frequency of the tone in Hz.
            duration_ms: Duration of the tone in milliseconds.
        """
        sample_rate = 44100
        n_samples = int(sample_rate * duration_ms / 1000.0)
        t = np.linspace(0, duration_ms / 1000.0, n_samples, endpoint=False)
        wave = (np.sin(2.0 * np.pi * freq * t) * 32767 * 0.5).astype(np.int16)

        # pygame.mixer expects a contiguous buffer
        sound = pygame.sndarray.make_sound(
            np.ascontiguousarray(wave.reshape(-1, 1))
        )
        sound.play()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _build_trial_sequence(self) -> List[str]:
        """Build a pseudo-randomised trial sequence.

        Trials are organised in blocks.  Each block contains one instance
        of every class, shuffled.  The total number of blocks equals
        ``n_trials_per_class`` so that each class appears exactly
        ``n_trials_per_class`` times.

        Returns:
            A flat list of class-name strings in presentation order.
        """
        sequence: List[str] = []
        for _ in range(self.n_trials_per_class):
            block = list(self.classes)
            random.shuffle(block)
            sequence.extend(block)
        return sequence

    def _timed_display(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        duration: float,
        draw_fn: Optional[callable] = None,
        progress_text: Optional[str] = None,
    ) -> None:
        """Show a display for a fixed duration, pumping the event queue.

        Args:
            screen: The pygame display surface.
            clock: A pygame Clock instance for frame timing.
            duration: How long to show this display, in seconds.
            draw_fn: Callable that draws on *screen*.  Called once at the
                start; the frame is held for *duration* seconds.
            progress_text: Optional progress indicator drawn in the
                bottom-right corner.
        """
        if draw_fn is not None:
            draw_fn(screen)
        if progress_text is not None:
            self._draw_progress(screen, progress_text)
        pygame.display.flip()

        start = time.time()
        while time.time() - start < duration:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                    return
            clock.tick(60)

    def _check_abort(self) -> bool:
        """Check whether the user has pressed ESC or closed the window.

        Returns:
            True if the paradigm should be aborted.
        """
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                return True
        return False

    def _show_break_screen(
        self,
        screen: pygame.Surface,
        clock: pygame.time.Clock,
        completed_run: int,
        total_runs: int,
    ) -> None:
        """Show a break screen and wait for SPACE to continue.

        Args:
            screen: The pygame display surface.
            clock: A pygame Clock instance.
            completed_run: The 1-based index of the run that just finished.
            total_runs: Total number of runs in the session.
        """
        screen.fill((0, 0, 0))
        w, h = screen.get_size()

        font_large = pygame.font.Font(None, 72)
        font_small = pygame.font.Font(None, 48)

        line1 = font_large.render(
            f"Run {completed_run} of {total_runs} complete",
            True,
            (255, 255, 255),
        )
        line2 = font_small.render(
            "Take a break. Press SPACE to continue.",
            True,
            (180, 180, 180),
        )

        screen.blit(line1, line1.get_rect(center=(w // 2, h // 2 - 40)))
        screen.blit(line2, line2.get_rect(center=(w // 2, h // 2 + 40)))
        pygame.display.flip()

        waiting = True
        while waiting:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        waiting = False
                    elif event.key == pygame.K_ESCAPE:
                        return
            clock.tick(30)

    # ------------------------------------------------------------------
    # Repr
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"GrazParadigm(classes={self.classes}, "
            f"trials_per_class={self.n_trials_per_class}, "
            f"runs={self.n_runs})"
        )
