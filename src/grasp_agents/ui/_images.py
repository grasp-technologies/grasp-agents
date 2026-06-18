"""
Inline-image zoom: the full-resolution modal and its terminal-graphics priming.

Inline images render as scroll-safe symbol-art (see :func:`._event_render`);
clicking one opens :class:`ImageZoomScreen`, the one place a graphics protocol
(TGP / Sixel) renders reliably. The priming helpers run the protocol/cell-size
probes once, before the app owns the screen.
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, ClassVar

from textual.app import ComposeResult
from textual.binding import BindingType
from textual.containers import Container
from textual.screen import ModalScreen
from textual.widgets import Static

from grasp_agents.types.content import InputImage

from ._event_render import image_to_pil, render_image, render_input_image

if TYPE_CHECKING:
    from textual.widget import Widget


def _prime_cell_size() -> None:
    # textual-image's halfcell/sixel/unicode renderers call get_cell_size() at
    # RENDER time; its `\x1b[16t` probe + stdin read then fights Textual for the
    # terminal once the app owns the screen → blank/garbled images (and stolen
    # keystrokes). The result is cached after the first call, so prime it now,
    # while the terminal is still in normal mode — the render path then returns
    # the cached value without ever probing. Best-effort.
    try:
        from textual_image._terminal import get_cell_size  # noqa: PLC0415, PLC2701

        get_cell_size()
    except Exception:
        pass


def prime_image_protocol() -> None:
    # The zoom modal renders full-res via textual-image's terminal-graphics
    # widget; its protocol + cell-size detection probes the terminal over stdin,
    # which fights Textual once the app owns the screen. Run it now, while the
    # terminal is still in normal mode — the results are cached for the modal.
    try:
        importlib.import_module("textual_image.widget")
    except Exception:
        pass
    _prime_cell_size()


if TYPE_CHECKING:

    def _zoom_widget(src: InputImage | str) -> Widget: ...

else:

    def _zoom_widget(src):
        # Full-resolution zoom view: textual-image's terminal-graphics Image
        # widget, auto-detecting the terminal's protocol (TGP on Kitty/Ghostty,
        # Sixel/unicode elsewhere). It lives on a dedicated, non-scrolling modal
        # screen — the one place graphics protocols are reliable, since nothing
        # scrolls the cells out from under them. Falls back to the inline
        # symbol-art renderable when the dep or terminal can't do graphics.
        pil = image_to_pil(src)
        if pil is not None:
            try:
                from textual_image.widget import Image as TImage  # noqa: PLC0415

                return TImage(pil, classes="zoom-img")
            except Exception:
                pass
        rend = (
            render_input_image(src)
            if isinstance(src, InputImage)
            else render_image(src)
        )
        return Static(rend, classes="zoom-img")


class ImageZoomScreen(ModalScreen[None]):
    """
    Full-screen overlay showing one image at full terminal-graphics fidelity.

    A dedicated, non-scrolling screen is the one place a graphics protocol (TGP
    on Kitty/Ghostty) renders reliably — nothing scrolls the cells out from
    under it. Dismissed with ``esc``/``q`` or a click anywhere.
    """

    # The image sizes itself (``width/height: auto``) inside a 90% box, so
    # textual-image fits it to the box preserving aspect ratio. Pinning the
    # widget to a fixed box instead would stretch the image to fill it.
    CSS = """
    ImageZoomScreen { align: center middle; background: $background 85%; }
    ImageZoomScreen #zoom-box { width: 90%; height: 90%; align: center middle; }
    ImageZoomScreen .zoom-img { width: auto; height: auto; }
    ImageZoomScreen #zoom-hint {
        dock: bottom; width: 1fr; text-align: center; color: $text-muted;
    }
    """

    BINDINGS: ClassVar[list[BindingType]] = [
        ("escape", "dismiss_zoom", "Close"),
        ("q", "dismiss_zoom", "Close"),
    ]

    def __init__(self, src: InputImage | str) -> None:
        super().__init__()
        self._src = src

    def compose(self) -> ComposeResult:
        with Container(id="zoom-box"):
            yield _zoom_widget(self._src)
        yield Static("esc / click to close", id="zoom-hint")

    def on_click(self) -> None:
        self.dismiss()

    def action_dismiss_zoom(self) -> None:
        self.dismiss()
