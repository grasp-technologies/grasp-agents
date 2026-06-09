"""
Notebook helpers for the TUI's event stream.

``render_events_inline`` displays events inline in a Jupyter cell using the
shared Rich renderers (and shows real images via IPython when available).
``screenshot`` / ``display_screenshot`` render the full multi-pane app headless
and return / embed it as an SVG — a static preview, since the live interactive
app needs a terminal.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from ._event_render import image_path_of, render_event

if TYPE_CHECKING:
    from collections.abc import AsyncIterator

    from ..types.events import Event

    # Typed shims: IPython is only partially typed, so the real calls live in
    # the runtime ``else`` branch (invisible to the type-checker).
    def _display_image(path: str) -> None: ...

    def _display_svg(svg: str) -> None: ...

else:

    def _display_image(path):
        from IPython.display import Image, display  # noqa: PLC0415

        display(Image(filename=path))

    def _display_svg(svg):
        from IPython.display import SVG, display  # noqa: PLC0415

        display(SVG(svg))


async def render_events_inline(
    stream: AsyncIterator[Event[Any]],
    *,
    console: Any = None,
    show_images: bool = True,
) -> AsyncIterator[Event[Any]]:
    """
    Display each event inline in a notebook cell; passthrough-yields events.

    Textual-free: uses only ``rich`` + the shared renderer. In a Jupyter
    context, a tool result carrying an ``image_path`` is also shown as a real
    image via ``IPython.display`` (not a half-block).
    """
    from rich.console import Console  # noqa: PLC0415

    console = console or Console()
    in_jupyter = bool(getattr(console, "is_jupyter", False))
    async for event in stream:
        renderable = render_event(event)
        if renderable is not None:
            console.print(renderable)
        if show_images and in_jupyter:
            path = image_path_of(event)
            if path is not None:
                _display_image(path)
        yield event


async def screenshot(
    events: AsyncIterator[Event[Any]],
    *,
    size: tuple[int, int] = (120, 40),
    title: str | None = None,
) -> str:
    """
    Render the full multi-pane app headless and return it as an SVG string.

    Captures the current (active) pane plus the agent tree — a static layout
    preview. Requires the ``tui`` extra (Textual).
    """
    from .app import GraspAgentsApp  # noqa: PLC0415

    app = GraspAgentsApp(events)
    async with app.run_test(size=size) as pilot:
        await app.wait_for_stream()
        await pilot.pause()
        return app.export_screenshot(title=title)


async def display_screenshot(
    events: AsyncIterator[Event[Any]],
    *,
    size: tuple[int, int] = (120, 40),
    title: str | None = None,
) -> None:
    """Render the app to an SVG and display it inline in a notebook."""
    _display_svg(await screenshot(events, size=size, title=title))
