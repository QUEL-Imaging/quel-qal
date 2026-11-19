import lazy_loader as lazy

__getattr__, __dir__, __all__ = lazy.attach_stub(__name__, __file__)


def _show_notebook_figures_inline() -> None:
    """Put ordinary notebook figures in the cell.

    Example notebooks import QAL before they plot. Doing this here means those
    notebooks do not need a Matplotlib magic. A script is not a notebook, so
    this leaves its window backend unchanged.
    """
    try:
        from qal.rta.roi_extraction.well_detector_gui import use_inline_plots

        use_inline_plots()
    except Exception:
        return


_show_notebook_figures_inline()