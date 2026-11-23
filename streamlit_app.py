"""Streamlit Cloud deployment entrypoint for CompressRoute.

Forwards execution to app/streamlit_app.py so both root 'streamlit_app.py'
and 'app/streamlit_app.py' paths work seamlessly on Streamlit Community Cloud.
"""

import os
import sys
import runpy

# Ensure repository root is on sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

# Execute the primary Streamlit application
app_path = os.path.join(root_dir, "app", "streamlit_app.py")
runpy.run_path(app_path, run_name="__main__")
