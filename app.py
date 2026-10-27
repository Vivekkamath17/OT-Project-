"""Root launcher for Streamlit app."""
import sys
from pathlib import Path

cloud_alloc_dir = Path(__file__).resolve().parent / "cloud_alloc"
sys.path.insert(0, str(cloud_alloc_dir))

from app import main

if __name__ == "__main__":
    main()
