"""AeroTwin — Physics-informed aircraft fuel-burn prediction."""

from importlib.metadata import version

try:
    __version__ = version("aerotwin")
except Exception:
    __version__ = "0.0.0"
