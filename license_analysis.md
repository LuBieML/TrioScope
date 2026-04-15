# Project License Analysis

This document analyzes the Python libraries used in the project and their respective licenses, as required to understand the conditions for distributing the application.

The core dependencies of this project are listed in `requirements.txt` and verified via source code inspection. Their respective licenses have been verified using package metadata (`pip show` and `.dist-info/METADATA`).

## Used Python Libraries and Licenses

### 1. PySide6
*   **Version:** >=6.5 (Current verified: 6.11.0)
*   **Purpose:** Python bindings for the Qt cross-platform application and UI framework. Used for building the Graphical User Interface.
*   **License:** `LGPL-3.0-only OR GPL-2.0-only OR GPL-3.0-only`
*   **Distribution Implications:**
    *   If you distribute the application under LGPLv3, you must dynamically link PySide6 (which happens by default with typical packaging tools like PyInstaller, but should be verified).
    *   You must provide end-users with the ability to modify the application to use a different version of the PySide6/Qt library.
    *   You must provide license notices indicating that PySide6 is used.
    *   If these conditions cannot be met (e.g., you want to statically link or keep the entire application tightly closed without allowing Qt replacement), a commercial license from The Qt Company would be required.

### 2. pyqtgraph
*   **Version:** >=0.13 (Current verified: 0.14.0)
*   **Purpose:** Scientific Graphics and GUI Library for Python. Used for rendering the real-time oscilloscope plots.
*   **License:** `MIT`
*   **Distribution Implications:** The MIT license is highly permissive. You can distribute this library with your application (even in a closed-source commercial product) as long as you include the original copyright notice and MIT license text.

### 3. PyOpenGL
*   **Version:** >=3.1 (Current verified: 3.1.10)
*   **Purpose:** Standard OpenGL bindings for Python. Used heavily by `pyqtgraph` for hardware-accelerated 3D/2D rendering.
*   **License:** `BSD License` (OSI Approved)
*   **Distribution Implications:** Similar to MIT, the BSD license is highly permissive. You can distribute this library as long as you retain the copyright notice and the disclaimer in the documentation/licenses provided with the distribution.

### 4. numpy
*   **Version:** >=1.24 (Current verified: 2.4.4)
*   **Purpose:** Fundamental package for array computing in Python. Used for efficient mathematical operations and handling large data arrays for the plots.
*   **License:** `BSD-3-Clause AND 0BSD AND MIT AND Zlib AND CC0-1.0`
*   **Distribution Implications:** Very permissive. You can distribute numpy freely as long as the license terms (which mostly require retaining copyright notices and license text) are respected in your distribution.

### 5. Trio_UnifiedApi
*   **Version:** >=1.2.0 (Current verified: 1.2.0)
*   **Purpose:** Python binding of the Trio Unified API. Used to communicate with the Trio Motion controllers.
*   **License:** `Other/Proprietary License` (Authored by Trio Motion Technology)
*   **Distribution Implications:** This is a proprietary library. You must consult the specific terms of the Trio Motion Technology software license agreement to determine the exact conditions under which you are allowed to distribute this library bundled with your application. Usually, this involves having an appropriate commercial agreement or using it exclusively with Trio hardware, but the explicit license terms provided by Trio must be followed.

## Summary for Distribution
To distribute this project legally:
1.  **Trio_UnifiedApi:** Ensure compliance with the proprietary license terms provided by Trio Motion Technology.
2.  **PySide6 (Qt):** Comply with the LGPLv3 obligations (dynamic linking, providing a way for users to swap the Qt library, providing license text) or acquire a commercial Qt license.
3.  **Permissive Libraries (pyqtgraph, PyOpenGL, numpy):** Include their respective copyright notices and license texts somewhere in the application's documentation or an "About" menu.
