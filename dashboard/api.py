"""
Flask API Backend for Harassment Detection Dashboard
- Serves data from MongoDB.
- Serves proof media files.
- Provides endpoints for the React frontend.
"""

import sys
import os
from flask import Flask, jsonify, send_from_directory
from flask_cors import CORS

# Ensure project root is on PYTHONPATH to import database_helper
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from database.database_helper import fetch_recent, fetch_stats

app = Flask(__name__)
CORS(app) # Enable CORS for all routes so React app can fetch

@app.route('/api/latest', methods=['GET'])
def get_latest():
    """Get the single most recent prediction for the live status banner."""
    records = fetch_recent(1)
    if records:
        return jsonify(records[0])
    return jsonify({"status": "no_data"})

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get recent prediction history (limit 50)."""
    records = fetch_recent(50)
    return jsonify(records)

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get aggregate statistics."""
    stats = fetch_stats()
    return jsonify(stats)

@app.route('/proof/<path:filename>')
def serve_proof(filename):
    """
    Serve proof files from proof/images/ and proof/audio/ directories.
    Filename should be something like 'images/image_101.jpg' or 'audio/audio_101.wav'
    """
    proof_dir = os.path.join(ROOT, "proof")
    return send_from_directory(proof_dir, filename)

if __name__ == '__main__':
    print("="*50)
    print(" Starting Flask API for Dashboard Backend")
    print(" Running on http://localhost:5000")
    print("="*50)
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
