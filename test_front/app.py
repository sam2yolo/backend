import json
import os
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

ACTIONS_FILE = os.path.join(os.path.dirname(__file__), "saved_actions.json")


def _load_actions():
    """Load saved actions from disk."""
    if not os.path.exists(ACTIONS_FILE):
        return []
    try:
        with open(ACTIONS_FILE, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return []


def _save_actions(actions):
    """Persist actions to disk."""
    with open(ACTIONS_FILE, "w") as f:
        json.dump(actions, f, indent=2)


@app.route("/")
def index():
    return render_template("index.html")


# -------- Saved Actions CRUD --------


@app.route("/api/actions", methods=["GET"])
def list_actions():
    actions = _load_actions()
    return jsonify(actions)


@app.route("/api/actions", methods=["POST"])
def create_action():
    data = request.get_json(silent=True)
    if not data or not data.get("action"):
        return jsonify({"error": "Missing 'action' field"}), 400

    actions = _load_actions()
    new_id = str(max([int(a.get("id", 0)) for a in actions] + [0]) + 1)
    entry = {
        "id": new_id,
        "action": data["action"],
        "payload": data.get("payload", {}),
        "label": data.get("label", data["action"]),
    }
    actions.append(entry)
    _save_actions(actions)
    return jsonify(entry), 201


@app.route("/api/actions/<action_id>", methods=["PUT"])
def update_action(action_id):
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "No data provided"}), 400

    actions = _load_actions()
    for entry in actions:
        if entry["id"] == action_id:
            if "action" in data:
                entry["action"] = data["action"]
            if "payload" in data:
                entry["payload"] = data["payload"]
            if "label" in data:
                entry["label"] = data["label"]
            _save_actions(actions)
            return jsonify(entry)
    return jsonify({"error": "Action not found"}), 404


@app.route("/api/actions/<action_id>", methods=["DELETE"])
def delete_action(action_id):
    actions = _load_actions()
    new_actions = [a for a in actions if a["id"] != action_id]
    if len(new_actions) == len(actions):
        return jsonify({"error": "Action not found"}), 404
    _save_actions(new_actions)
    return jsonify({"status": "deleted"})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
