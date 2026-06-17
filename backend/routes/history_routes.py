"""
backend/routes/history_routes.py
-----------------------------------
"My History" endpoints — per-user activity history.
  GET    /api/v1/history            — list the logged-in user's history (paginated)
  GET    /api/v1/history/<id>       — reopen one history entry (full payload + result)
  DELETE /api/v1/history/<id>       — delete one of the logged-in user's history entries

Every entry is scoped to the authenticated user; a user can only ever see or
delete their own history.
"""
from __future__ import annotations

import logging

from flask import Blueprint, jsonify, request

from backend.auth.dependencies   import require_auth
from backend.models.schemas      import validate_pagination, ok, err
from backend.services.db_service import (
    list_history, get_history_entry, delete_history_entry, save_history,
)

logger = logging.getLogger(__name__)
bp     = Blueprint("history", __name__, url_prefix="/api/v1/history")

_VALID_ENTRY_TYPES = {
    "single_prediction", "batch_prediction", "groq_prediction",
    "groq_batch", "report", "analysis",
}


# ── POST /history ──────────────────────────────────────────────────────────────
@bp.route("", methods=["POST"])
@require_auth
def create_history_entry(current_user):
    """
    Record a history entry for the logged-in user (e.g. a generated report
    or analysis that was produced client-side and has no other save path).
    ---
    tags: [History]
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            required: [entry_type, title]
            properties:
              entry_type: {type: string, enum: [single_prediction, batch_prediction, groq_prediction, groq_batch, report, analysis]}
              title:      {type: string}
              summary:    {type: string}
              risk_category: {type: string}
              payload:    {type: object}
              result:     {type: object}
    responses:
      201:
        description: History entry created
      400:
        description: Validation error
    """
    data = request.get_json(silent=True) or {}
    entry_type = str(data.get("entry_type", "")).strip()
    title      = str(data.get("title", "")).strip()

    if entry_type not in _VALID_ENTRY_TYPES:
        return jsonify(err(f"'entry_type' must be one of {sorted(_VALID_ENTRY_TYPES)}.",
                            "VALIDATION_ERROR")[0]), 400
    if not title:
        return jsonify(err("'title' is required.", "VALIDATION_ERROR")[0]), 400

    history_id = save_history(
        user_id=current_user["id"],
        entry_type=entry_type,
        title=title,
        summary=data.get("summary"),
        risk_category=data.get("risk_category"),
        payload=data.get("payload"),
        result=data.get("result"),
    )
    return jsonify(ok({"history_id": history_id}, "History entry saved.")[0]), 201


# ── GET /history ───────────────────────────────────────────────────────────────
@bp.route("", methods=["GET"])
@require_auth
def list_my_history(current_user):
    """
    List the authenticated user's past predictions / batches / reports.
    ---
    tags: [History]
    security:
      - BearerAuth: []
    parameters:
      - in: query
        name: page
        schema: {type: integer, default: 1}
      - in: query
        name: page_size
        schema: {type: integer, default: 20}
      - in: query
        name: entry_type
        schema: {type: string, enum: [single_prediction, batch_prediction, groq_prediction, groq_batch, report, analysis]}
    responses:
      200:
        description: Paginated list of the user's own history entries
    """
    page, page_size = validate_pagination(request.args)
    entry_type = request.args.get("entry_type")
    result = list_history(current_user["id"], page, page_size, entry_type)
    return jsonify(ok(result)[0]), 200


# ── GET /history/<id> ─────────────────────────────────────────────────────────
@bp.route("/<int:history_id>", methods=["GET"])
@require_auth
def reopen_history(history_id: int, current_user):
    """
    Reopen one history entry, including the original payload and result.
    ---
    tags: [History]
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: history_id
        required: true
        schema: {type: integer}
    responses:
      200:
        description: Full history entry
      404:
        description: Not found or not owned by the current user
    """
    entry = get_history_entry(current_user["id"], history_id)
    if not entry:
        return jsonify(err("History entry not found.", "NOT_FOUND")[0]), 404
    return jsonify(ok(entry)[0]), 200


# ── DELETE /history/<id> ──────────────────────────────────────────────────────
@bp.route("/<int:history_id>", methods=["DELETE"])
@require_auth
def remove_history(history_id: int, current_user):
    """
    Delete one of the authenticated user's own history entries.
    ---
    tags: [History]
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: history_id
        required: true
        schema: {type: integer}
    responses:
      200:
        description: Deleted successfully
      404:
        description: Not found or not owned by the current user
    """
    deleted = delete_history_entry(current_user["id"], history_id)
    if not deleted:
        return jsonify(err("History entry not found.", "NOT_FOUND")[0]), 404
    return jsonify(ok(None, "History entry deleted.")[0]), 200
