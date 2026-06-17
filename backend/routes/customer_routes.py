"""
backend/routes/customer_routes.py
-----------------------------------
Customer management endpoints:
  GET    /api/v1/customers            — list with pagination + risk filter
  POST   /api/v1/customers            — create customer record
  GET    /api/v1/customers/<id>       — single customer with latest prediction
  PUT    /api/v1/customers/<id>       — update customer
  GET    /api/v1/customers/<id>/history — prediction history
"""
from __future__ import annotations
import logging
from flask import Blueprint, jsonify, request
from backend.auth.dependencies   import require_auth, require_role
from backend.models.schemas      import validate_customer, validate_pagination, ok, err
from backend.services.db_service import (
    create_customer, get_customer, list_customers, update_customer,
    get_prediction_history, get_latest_prediction, get_recommendations,
)

logger = logging.getLogger(__name__)
bp     = Blueprint("customers", __name__, url_prefix="/api/v1/customers")


@bp.route("", methods=["GET"])
@require_auth
def list_all(current_user):
    """
    List all customers with optional risk filter.
    ---
    tags: [Customers]
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
        name: risk
        schema: {type: string, enum: [High, Medium, Low]}
        description: Filter by latest prediction risk category
    responses:
      200:
        description: Paginated customer list
    """
    page, page_size = validate_pagination(request.args)
    risk_filter = request.args.get("risk")
    result = list_customers(page, page_size, risk_filter)
    return jsonify(ok(result)[0]), 200


@bp.route("", methods=["POST"])
@require_auth
def create(current_user):
    """
    Create a new customer record.
    ---
    tags: [Customers]
    security:
      - BearerAuth: []
    requestBody:
      required: true
      content:
        application/json:
          schema:
            $ref: '#/components/schemas/CustomerInput'
    responses:
      201:
        description: Customer created
      400:
        description: Validation error
    """
    data = request.get_json(silent=True) or {}
    try:
        clean = validate_customer(data)
    except ValueError as e:
        return jsonify(err(str(e), "VALIDATION_ERROR")[0]), 400
    customer = create_customer(clean)
    return jsonify(ok(customer, "Customer created.")[0]), 201


@bp.route("/<int:customer_id>", methods=["GET"])
@require_auth
def detail(customer_id: int, current_user):
    """
    Get a customer by ID with their latest prediction and recommendations.
    ---
    tags: [Customers]
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: customer_id
        required: true
        schema: {type: integer}
    responses:
      200:
        description: Customer detail
      404:
        description: Not found
    """
    customer = get_customer(customer_id)
    if not customer:
        return jsonify(err("Customer not found.", "NOT_FOUND")[0]), 404

    latest_pred = get_latest_prediction(customer_id)
    recommendations = get_recommendations(customer_id)

    return jsonify(ok({
        "customer":        customer,
        "latest_prediction": latest_pred,
        "recommendations": recommendations,
    })[0]), 200


@bp.route("/<int:customer_id>", methods=["PUT", "PATCH"])
@require_auth
def update(customer_id: int, current_user):
    """
    Update customer attributes.
    ---
    tags: [Customers]
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: customer_id
        required: true
        schema: {type: integer}
    requestBody:
      required: true
      content:
        application/json:
          schema:
            type: object
            properties:
              tenure:            {type: integer}
              monthly_charges:   {type: number}
              contract:          {type: string}
              payment_method:    {type: string}
              inactive_days:     {type: integer}
    responses:
      200:
        description: Customer updated
      404:
        description: Not found
    """
    if not get_customer(customer_id):
        return jsonify(err("Customer not found.", "NOT_FOUND")[0]), 404
    data    = request.get_json(silent=True) or {}
    updated = update_customer(customer_id, data)
    return jsonify(ok(updated, "Customer updated.")[0]), 200


@bp.route("/<int:customer_id>/history", methods=["GET"])
@require_auth
def history(customer_id: int, current_user):
    """
    Return full prediction history for a customer.
    ---
    tags: [Customers]
    security:
      - BearerAuth: []
    parameters:
      - in: path
        name: customer_id
        required: true
        schema: {type: integer}
    responses:
      200:
        description: List of past predictions
      404:
        description: Customer not found
    """
    if not get_customer(customer_id):
        return jsonify(err("Customer not found.", "NOT_FOUND")[0]), 404
    hist = get_prediction_history(customer_id)
    return jsonify(ok({"customer_id": customer_id, "predictions": hist,
                       "count": len(hist)})[0]), 200
