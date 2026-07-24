"""
Backward-compatible entry alias.
Prefer: ``python app.py``
"""
from app import main

if __name__ == "__main__":
    main()
