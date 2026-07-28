"""Minimal build-time compatibility shim for Torch's Detectron2 extension build.

The target py310 environment uses a setuptools release that no longer ships
``pkg_resources``.  Torch 2.0 imports only its ``packaging`` attribute while
building C++ extensions; this local shim avoids downgrading any installed
library and is used only through PYTHONPATH during the Detectron2 build.
"""

import packaging

