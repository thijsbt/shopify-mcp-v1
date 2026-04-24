#!/usr/bin/env python3
"""
Shopify MCP Server — Full Admin API access via FastMCP.
Provides tools for managing products, orders, customers, collections,
inventory, and fulfillments through the Shopify Admin REST API.

Token Management:
  - Uses client_credentials grant to auto-generate and refresh tokens
  - Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET (recommended for OAuth apps)
  - Falls back to static SHOPIFY_ACCESS_TOKEN if client credentials not set
"""
import json
import os
import logging
import time
import asyncio
from typing import Optional, List, Dict, Any
from enum import Enum
import httpx
from pydantic import BaseModel, Field, ConfigDict, field_validator
from mcp.server.fastmcp import FastMCP

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SHOPIFY_STORE        = os.environ.get("SHOPIFY_STORE", "")           # e.g. "my-store"
SHOPIFY_TOKEN        = os.environ.get("SHOPIFY_ACCESS_TOKEN", "")    # Static token (shpat_...)
SHOPIFY_CLIENT_ID    = os.environ.get("SHOPIFY_CLIENT_ID", "")
SHOPIFY_CLIENT_SECRET = os.environ.get("SHOPIFY_CLIENT_SECRET", "")
API_VERSION          = os.environ.get("SHOPIFY_API_VERSION", "2024-10")

# Refresh buffer: refresh token 30 minutes before expiry (only used with OAuth)
TOKEN_REFRESH_BUFFER = int(os.environ.get("TOKEN_REFRESH_BUFFER", "1800"))

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("shopify_mcp")

PORT          = int(os.environ.get("PORT", "8000"))
MCP_TRANSPORT = os.environ.get("MCP_TRANSPORT", "streamable-http")

mcp = FastMCP("shopify_mcp", host="0.0.0.0", port=PORT, json_response=True)


# ---------------------------------------------------------------------------
# Token Manager — handles automatic token lifecycle
# ---------------------------------------------------------------------------

class TokenManager:
    """
    Manages Shopify Admin API access tokens.

    Two modes:
      1. Static token  — set SHOPIFY_ACCESS_TOKEN (recommended for Custom Apps)
      2. OAuth / client_credentials — set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET
         Enables auto-refresh before expiry and retry on 401.
    """

    def __init__(
        self,
        store: str,
        client_id: str,
        client_secret: str,
        static_token: str = "",
        refresh_buffer: int = 1800,
    ):
        self._store         = store
        self._client_id     = client_id
        self._client_secret = client_secret
        self._static_token  = static_token
        self._refresh_buffer = refresh_buffer

        self._access_token: str   = ""
        self._expires_at: float   = 0.0
        self._lock = asyncio.Lock()

        self._use_client_credentials = bool(client_id and client_secret)

        if self._use_client_credentials:
            logger.info("Token mode: client_credentials (auto-refresh enabled)")
        elif static_token:
            logger.info("Token mode: static SHOPIFY_ACCESS_TOKEN (no auto-refresh)")
            self._access_token = static_token
            self._expires_at   = float("inf")
        else:
            logger.warning(
                "No credentials configured. Set SHOPIFY_ACCESS_TOKEN or "
                "SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET."
            )

    @property
    def is_expired(self) -> bool:
        if not self._access_token:
            return True
        return time.time() >= (self._expires_at - self._refresh_buffer)

    async def get_token(self) -> str:
        if not self.is_expired:
            return self._access_token

        async with self._lock:
            if not self.is_expired:
                return self._access_token

            if self._use_client_credentials:
                await self._refresh_token()
            elif not self._access_token:
                raise RuntimeError(
                    "No valid token available. "
                    "Set SHOPIFY_ACCESS_TOKEN in your environment variables."
                )

        return self._access_token

    async def force_refresh(self) -> str:
        if not self._use_client_credentials:
            raise RuntimeError(
                "Cannot refresh — using a static token. "
                "Set SHOPIFY_CLIENT_ID + SHOPIFY_CLIENT_SECRET to enable auto-refresh."
            )
        async with self._lock:
            await self._refresh_token()
        return self._access_token

    async def _refresh_token(self) -> None:
        url = f"https://{self._store}.myshopify.com/admin/oauth/access_token"
        logger.info("Refreshing Shopify access token via client_credentials grant...")

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                url,
                data={
                    "grant_type":    "client_credentials",
                    "client_id":     self._client_id,
                    "client_secret": self._client_secret,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
                timeout=15.0,
            )

            if resp.status_code != 200:
                logger.error(f"Token refresh failed ({resp.status_code}): {resp.text[:500]}")
                raise RuntimeError(
                    f"Token refresh failed ({resp.status_code}). "
                    "Check SHOPIFY_CLIENT_ID and SHOPIFY_CLIENT_SECRET."
                )

            data               = resp.json()
            self._access_token = data["access_token"]
            expires_in         = data.get("expires_in", 86399)
            self._expires_at   = time.time() + expires_in

            scope         = data.get("scope", "")
            scope_preview = scope[:80] + "..." if len(scope) > 80 else scope
            logger.info(
                f"Token refreshed. Expires in {expires_in}s "
                f"({expires_in // 3600}h {(expires_in % 3600) // 60}m). "
                f"Scopes: {scope_preview}"
            )


# Global token manager
token_manager = TokenManager(
    store=SHOPIFY_STORE,
    client_id=SHOPIFY_CLIENT_ID,
    client_secret=SHOPIFY_CLIENT_SECRET,
    static_token=SHOPIFY_TOKEN,
    refresh_buffer=TOKEN_REFRESH_BUFFER,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _base_url() -> str:
    return f"https://{SHOPIFY_STORE}.myshopify.com/admin/api/{API_VERSION}"


async def _headers() -> dict:
    token = await token_manager.get_token()
    return {
        "X-Shopify-Access-Token": token,
        "Content-Type": "application/json",
    }


async def _request(
    method: str,
    path: str,
    params: Optional[dict] = None,
    body:   Optional[dict] = None,
    _retried: bool = False,
) -> dict:
    """Central HTTP helper — every API call flows through here.
    Auto-retries once on 401 when using OAuth credentials.
    """
    if not SHOPIFY_STORE:
        raise RuntimeError(
            "Missing SHOPIFY_STORE environment variable. "
            "Set it before starting the server."
        )

    url     = f"{_base_url()}/{path}"
    headers = await _headers()

    async with httpx.AsyncClient() as client:
        resp = await client.request(
            method, url,
            headers=headers,
            params=params,
            json=body,
            timeout=30.0,
        )

        if resp.status_code == 401 and not _retried and token_manager._use_client_credentials:
            logger.warning("Got 401 — refreshing token and retrying...")
            await token_manager.force_refresh()
            return await _request(method, path, params=params, body=body, _retried=True)

        resp.raise_for_status()
        if resp.status_code == 204:
            return {}
        return resp.json()


def _error(e: Exception) -> str:
    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        try:
            detail = e.response.json()
        except Exception:
            detail = e.response.text[:500]
        messages = {
            401: "Authentication failed — check your SHOPIFY_ACCESS_TOKEN (should start with shpat_).",
            403: "Permission denied — your token may be missing required API scopes.",
            404: "Resource not found — double-check the ID.",
            422: f"Validation error: {json.dumps(detail)}",
            429: "Rate-limited — wait a moment and retry.",
        }
        return messages.get(status, f"Shopify API error {status}: {json.dumps(detail)}")
    if isinstance(e, httpx.TimeoutException):
        return "Request timed out — try again."
    if isinstance(e, RuntimeError):
        return str(e)
    return f"Unexpected error: {type(e).__name__}: {e}"


def _fmt(data: Any) -> str:
    return json.dumps(data, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCTS
# ═══════════════════════════════════════════════════════════════════════════

class ListProductsInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int]  = Field(default=50, ge=1, le=250, description="Max products to return (1-250)")
    status:         Optional[str]  = Field(default=None, description="Filter by status: active, archived, draft")
    product_type:   Optional[str]  = Field(default=None, description="Filter by product type")
    vendor:         Optional[str]  = Field(default=None, description="Filter by vendor name")
    collection_id:  Optional[int]  = Field(default=None, description="Filter by collection ID")
    since_id:       Optional[int]  = Field(default=None, description="Pagination: return products after this ID")
    fields:         Optional[str]  = Field(default=None, description="Comma-separated fields to include")


@mcp.tool(
    name="shopify_list_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_products(params: ListProductsInput) -> str:
    """List products from the Shopify store with optional filters."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["status", "product_type", "vendor", "collection_id", "since_id", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data     = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


class GetProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="The Shopify product ID")


@mcp.tool(
    name="shopify_get_product",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_product(params: GetProductInput) -> str:
    """Retrieve a single product by ID, including all variants and images."""
    try:
        data = await _request("GET", f"products/{params.product_id}.json")
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class CreateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:        str                        = Field(..., min_length=1, description="Product title")
    body_html:    Optional[str]              = Field(default=None, description="HTML description")
    vendor:       Optional[str]              = Field(default=None)
    product_type: Optional[str]              = Field(default=None)
    tags:         Optional[str]              = Field(default=None, description="Comma-separated tags")
    status:       Optional[str]              = Field(default="draft", description="active, archived, or draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None, description="Variant objects with price, sku, etc.")
    options:      Optional[List[Dict[str, Any]]] = Field(default=None, description="Product options (Size, Color, etc.)")
    images:       Optional[List[Dict[str, Any]]] = Field(default=None, description="Image objects with src URL")


@mcp.tool(
    name="shopify_create_product",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_product(params: CreateProductInput) -> str:
    """Create a new product in the Shopify store."""
    try:
        product: Dict[str, Any] = {"title": params.title}
        for field in ["body_html", "vendor", "product_type", "tags", "status", "variants", "options", "images"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("POST", "products.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class UpdateProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:   int            = Field(..., description="Product ID to update")
    title:        Optional[str]  = Field(default=None)
    body_html:    Optional[str]  = Field(default=None)
    vendor:       Optional[str]  = Field(default=None)
    product_type: Optional[str]  = Field(default=None)
    tags:         Optional[str]  = Field(default=None)
    status:       Optional[str]  = Field(default=None, description="active, archived, or draft")
    variants:     Optional[List[Dict[str, Any]]] = Field(default=None)


@mcp.tool(
    name="shopify_update_product",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_product(params: UpdateProductInput) -> str:
    """Update an existing product. Only provided fields are changed."""
    try:
        product: Dict[str, Any] = {}
        for field in ["title", "body_html", "vendor", "product_type", "tags", "status", "variants"]:
            val = getattr(params, field)
            if val is not None:
                product[field] = val
        data = await _request("PUT", f"products/{params.product_id}.json", body={"product": product})
        return _fmt(data.get("product", data))
    except Exception as e:
        return _error(e)


class DeleteProductInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID to delete")


@mcp.tool(
    name="shopify_delete_product",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_product(params: DeleteProductInput) -> str:
    """Permanently delete a product. This cannot be undone."""
    try:
        await _request("DELETE", f"products/{params.product_id}.json")
        return f"Product {params.product_id} deleted."
    except Exception as e:
        return _error(e)


class ProductCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:       Optional[str] = Field(default=None, description="active, archived, or draft")
    vendor:       Optional[str] = Field(default=None)
    product_type: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_count_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_count_products(params: ProductCountInput) -> str:
    """Get the total count of products, optionally filtered."""
    try:
        p: Dict[str, Any] = {}
        for field in ["status", "vendor", "product_type"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "products/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# ORDERS
# ═══════════════════════════════════════════════════════════════════════════

class ListOrdersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:               Optional[int] = Field(default=50, ge=1, le=250)
    status:              Optional[str] = Field(default="any", description="open, closed, cancelled, any")
    financial_status:    Optional[str] = Field(default=None, description="authorized, pending, paid, refunded, voided, any")
    fulfillment_status:  Optional[str] = Field(default=None, description="shipped, partial, unshipped, unfulfilled, any")
    since_id:            Optional[int] = Field(default=None)
    created_at_min:      Optional[str] = Field(default=None, description="ISO 8601 date, e.g. 2024-01-01T00:00:00Z")
    created_at_max:      Optional[str] = Field(default=None)
    fields:              Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_orders(params: ListOrdersInput) -> str:
    """List orders with optional filters for status, financial/fulfillment status, and date range."""
    try:
        p: Dict[str, Any] = {"limit": params.limit, "status": params.status}
        for field in ["financial_status", "fulfillment_status", "since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data   = await _request("GET", "orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


class GetOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="The Shopify order ID")


@mcp.tool(
    name="shopify_get_order",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_order(params: GetOrderInput) -> str:
    """Retrieve a single order by ID with full details."""
    try:
        data = await _request("GET", f"orders/{params.order_id}.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


class OrderCountInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    status:             Optional[str] = Field(default="any")
    financial_status:   Optional[str] = Field(default=None)
    fulfillment_status: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_count_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_count_orders(params: OrderCountInput) -> str:
    """Get total order count, optionally filtered."""
    try:
        p: Dict[str, Any] = {"status": params.status}
        for field in ["financial_status", "fulfillment_status"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data = await _request("GET", "orders/count.json", params=p)
        return _fmt(data)
    except Exception as e:
        return _error(e)


class CloseOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int = Field(..., description="Order ID to close")


@mcp.tool(
    name="shopify_close_order",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_close_order(params: CloseOrderInput) -> str:
    """Close an order (marks it as completed)."""
    try:
        data = await _request("POST", f"orders/{params.order_id}/close.json")
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


class CancelOrderInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int            = Field(..., description="Order ID to cancel")
    reason:   Optional[str]  = Field(default=None, description="customer, fraud, inventory, declined, other")
    email:    Optional[bool] = Field(default=True,  description="Send cancellation email to customer")
    restock:  Optional[bool] = Field(default=False, description="Restock line items")


@mcp.tool(
    name="shopify_cancel_order",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_cancel_order(params: CancelOrderInput) -> str:
    """Cancel an order. Optionally restock items and notify the customer."""
    try:
        body: Dict[str, Any] = {}
        for field in ["reason", "email", "restock"]:
            val = getattr(params, field)
            if val is not None:
                body[field] = val
        data = await _request("POST", f"orders/{params.order_id}/cancel.json", body=body)
        return _fmt(data.get("order", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# CUSTOMERS
# ═══════════════════════════════════════════════════════════════════════════

class ListCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    limit:          Optional[int] = Field(default=50, ge=1, le=250)
    since_id:       Optional[int] = Field(default=None)
    created_at_min: Optional[str] = Field(default=None, description="ISO 8601 date")
    created_at_max: Optional[str] = Field(default=None)
    fields:         Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_customers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_customers(params: ListCustomersInput) -> str:
    """List customers from the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for f in ["since_id", "created_at_min", "created_at_max", "fields"]:
            val = getattr(params, f)
            if val is not None:
                p[f] = val
        data      = await _request("GET", "customers.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)


class SearchCustomersInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    query: str           = Field(..., min_length=1, description="Search query (name, email, etc.)")
    limit: Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_search_customers",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_search_customers(params: SearchCustomersInput) -> str:
    """Search customers by name, email, or other fields."""
    try:
        p         = {"query": params.query, "limit": params.limit}
        data      = await _request("GET", "customers/search.json", params=p)
        customers = data.get("customers", [])
        return _fmt({"count": len(customers), "customers": customers})
    except Exception as e:
        return _error(e)


class GetCustomerInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int = Field(..., description="Shopify customer ID")


@mcp.tool(
    name="shopify_get_customer",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_customer(params: GetCustomerInput) -> str:
    """Retrieve a single customer by ID."""
    try:
        data = await _request("GET", f"customers/{params.customer_id}.json")
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class CreateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    first_name:         Optional[str]  = Field(default=None)
    last_name:          Optional[str]  = Field(default=None)
    email:              Optional[str]  = Field(default=None)
    phone:              Optional[str]  = Field(default=None)
    tags:               Optional[str]  = Field(default=None)
    note:               Optional[str]  = Field(default=None)
    addresses:          Optional[List[Dict[str, Any]]] = Field(default=None)
    send_email_invite:  Optional[bool] = Field(default=False)


@mcp.tool(
    name="shopify_create_customer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_customer(params: CreateCustomerInput) -> str:
    """Create a new customer."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note", "addresses", "send_email_invite"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("POST", "customers.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class UpdateCustomerInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    customer_id: int           = Field(..., description="Customer ID to update")
    first_name:  Optional[str] = Field(default=None)
    last_name:   Optional[str] = Field(default=None)
    email:       Optional[str] = Field(default=None)
    phone:       Optional[str] = Field(default=None)
    tags:        Optional[str] = Field(default=None)
    note:        Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_customer",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_customer(params: UpdateCustomerInput) -> str:
    """Update an existing customer. Only provided fields are changed."""
    try:
        customer: Dict[str, Any] = {}
        for field in ["first_name", "last_name", "email", "phone", "tags", "note"]:
            val = getattr(params, field)
            if val is not None:
                customer[field] = val
        data = await _request("PUT", f"customers/{params.customer_id}.json", body={"customer": customer})
        return _fmt(data.get("customer", data))
    except Exception as e:
        return _error(e)


class CustomerOrdersInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    customer_id: int           = Field(..., description="Customer ID")
    limit:       Optional[int] = Field(default=50, ge=1, le=250)
    status:      Optional[str] = Field(default="any")


@mcp.tool(
    name="shopify_get_customer_orders",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_customer_orders(params: CustomerOrdersInput) -> str:
    """Get all orders for a specific customer."""
    try:
        p      = {"limit": params.limit, "status": params.status}
        data   = await _request("GET", f"customers/{params.customer_id}/orders.json", params=p)
        orders = data.get("orders", [])
        return _fmt({"count": len(orders), "orders": orders})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# COLLECTIONS (Custom + Smart)
# ═══════════════════════════════════════════════════════════════════════════

class ListCollectionsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:           Optional[int] = Field(default=50, ge=1, le=250)
    since_id:        Optional[int] = Field(default=None)
    collection_type: Optional[str] = Field(default="custom", description="'custom' or 'smart'")


@mcp.tool(
    name="shopify_list_collections",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_collections(params: ListCollectionsInput) -> str:
    """List custom or smart collections."""
    try:
        endpoint = "custom_collections.json" if params.collection_type == "custom" else "smart_collections.json"
        p: Dict[str, Any] = {"limit": params.limit}
        if params.since_id:
            p["since_id"] = params.since_id
        data = await _request("GET", endpoint, params=p)
        key  = "custom_collections" if params.collection_type == "custom" else "smart_collections"
        collections = data.get(key, [])
        return _fmt({"count": len(collections), "collections": collections})
    except Exception as e:
        return _error(e)


class GetCollectionProductsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: int           = Field(..., description="Collection ID")
    limit:         Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_get_collection_products",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_collection_products(params: GetCollectionProductsInput) -> str:
    """Get all products in a specific collection."""
    try:
        p        = {"limit": params.limit, "collection_id": params.collection_id}
        data     = await _request("GET", "products.json", params=p)
        products = data.get("products", [])
        return _fmt({"count": len(products), "products": products})
    except Exception as e:
        return _error(e)


class GetCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id:   int           = Field(..., description="Collection ID")
    collection_type: Optional[str] = Field(default="custom", description="'custom' or 'smart'")


@mcp.tool(
    name="shopify_get_collection",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_collection(params: GetCollectionInput) -> str:
    """Retrieve a single collection by ID (custom or smart)."""
    try:
        endpoint = f"custom_collections/{params.collection_id}.json" if params.collection_type == "custom" else f"smart_collections/{params.collection_id}.json"
        key      = "custom_collection" if params.collection_type == "custom" else "smart_collection"
        data     = await _request("GET", endpoint)
        return _fmt(data.get(key, data))
    except Exception as e:
        return _error(e)


class CreateCustomCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:           str                   = Field(..., min_length=1, description="Collection title")
    body_html:       Optional[str]         = Field(default=None, description="HTML description")
    handle:          Optional[str]         = Field(default=None, description="URL handle (slug). Auto-generated from title if omitted")
    published:       Optional[bool]        = Field(default=False, description="Publish to online store. Default False (hidden)")
    sort_order:      Optional[str]         = Field(default=None, description="alpha-asc, alpha-desc, best-selling, created, created-desc, manual, price-asc, price-desc")
    image_src:       Optional[str]         = Field(default=None, description="Public URL of collection image")
    product_ids:     Optional[List[int]]   = Field(default=None, description="Product IDs to include (manual curation)")
    metafields:      Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline metafields (e.g. SEO title/description)")


@mcp.tool(
    name="shopify_create_custom_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_custom_collection(params: CreateCustomCollectionInput) -> str:
    """Create a manually curated (custom) collection where products are added by ID."""
    try:
        collection: Dict[str, Any] = {"title": params.title, "published": bool(params.published)}
        if params.body_html is not None:
            collection["body_html"] = params.body_html
        if params.handle:
            collection["handle"] = params.handle
        if params.sort_order:
            collection["sort_order"] = params.sort_order
        if params.image_src:
            collection["image"] = {"src": params.image_src}
        if params.product_ids:
            collection["collects"] = [{"product_id": pid} for pid in params.product_ids]
        if params.metafields:
            collection["metafields"] = params.metafields
        data = await _request("POST", "custom_collections.json", body={"custom_collection": collection})
        return _fmt(data.get("custom_collection", data))
    except Exception as e:
        return _error(e)


class SmartCollectionRule(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    column:    str = Field(..., description="tag, title, type, vendor, variant_price, variant_compare_at_price, variant_weight, variant_inventory, variant_title")
    relation:  str = Field(..., description="equals, not_equals, greater_than, less_than, starts_with, ends_with, contains, not_contains")
    condition: str = Field(..., description="The value to match, e.g. 'graduation-dress' for a tag rule")


class CreateSmartCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:       str                         = Field(..., min_length=1)
    rules:       List[SmartCollectionRule]   = Field(..., min_length=1, description="Auto-membership rules")
    disjunctive: Optional[bool]              = Field(default=False, description="False = AND (all rules), True = OR (any rule)")
    body_html:   Optional[str]               = Field(default=None)
    handle:      Optional[str]               = Field(default=None)
    published:   Optional[bool]              = Field(default=False, description="Publish to online store. Default False (hidden)")
    sort_order:  Optional[str]               = Field(default=None)
    image_src:   Optional[str]               = Field(default=None)
    metafields:  Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline metafields (e.g. SEO title/description)")


@mcp.tool(
    name="shopify_create_smart_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_smart_collection(params: CreateSmartCollectionInput) -> str:
    """Create a smart collection with auto-membership rules (e.g. tag = 'graduation-dress')."""
    try:
        collection: Dict[str, Any] = {
            "title":       params.title,
            "rules":       [r.model_dump() for r in params.rules],
            "disjunctive": bool(params.disjunctive),
            "published":   bool(params.published),
        }
        if params.body_html is not None:
            collection["body_html"] = params.body_html
        if params.handle:
            collection["handle"] = params.handle
        if params.sort_order:
            collection["sort_order"] = params.sort_order
        if params.image_src:
            collection["image"] = {"src": params.image_src}
        if params.metafields:
            collection["metafields"] = params.metafields
        data = await _request("POST", "smart_collections.json", body={"smart_collection": collection})
        return _fmt(data.get("smart_collection", data))
    except Exception as e:
        return _error(e)


class UpdateCollectionInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    collection_id:   int                          = Field(..., description="Collection ID")
    collection_type: Optional[str]                = Field(default="custom", description="'custom' or 'smart'")
    title:           Optional[str]                = Field(default=None)
    body_html:       Optional[str]                = Field(default=None)
    handle:          Optional[str]                = Field(default=None)
    published:       Optional[bool]               = Field(default=None)
    sort_order:      Optional[str]                = Field(default=None)
    image_src:       Optional[str]                = Field(default=None)
    rules:           Optional[List[SmartCollectionRule]] = Field(default=None, description="(Smart only) Replace rules")
    disjunctive:     Optional[bool]               = Field(default=None, description="(Smart only) AND vs OR")


@mcp.tool(
    name="shopify_update_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_collection(params: UpdateCollectionInput) -> str:
    """Update an existing collection. Only provided fields are changed."""
    try:
        is_smart  = params.collection_type == "smart"
        endpoint  = f"smart_collections/{params.collection_id}.json" if is_smart else f"custom_collections/{params.collection_id}.json"
        body_key  = "smart_collection" if is_smart else "custom_collection"

        payload: Dict[str, Any] = {"id": params.collection_id}
        for field in ["title", "body_html", "handle", "sort_order"]:
            val = getattr(params, field)
            if val is not None:
                payload[field] = val
        if params.published is not None:
            payload["published"] = params.published
        if params.image_src:
            payload["image"] = {"src": params.image_src}
        if is_smart and params.rules is not None:
            payload["rules"] = [r.model_dump() for r in params.rules]
        if is_smart and params.disjunctive is not None:
            payload["disjunctive"] = params.disjunctive

        data = await _request("PUT", endpoint, body={body_key: payload})
        return _fmt(data.get(body_key, data))
    except Exception as e:
        return _error(e)


class DeleteCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id:   int           = Field(..., description="Collection ID")
    collection_type: Optional[str] = Field(default="custom", description="'custom' or 'smart'")


@mcp.tool(
    name="shopify_delete_collection",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_collection(params: DeleteCollectionInput) -> str:
    """Permanently delete a collection. This cannot be undone."""
    try:
        endpoint = f"smart_collections/{params.collection_id}.json" if params.collection_type == "smart" else f"custom_collections/{params.collection_id}.json"
        await _request("DELETE", endpoint)
        return _fmt({"deleted": True, "collection_id": params.collection_id})
    except Exception as e:
        return _error(e)


class AddProductToCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: int           = Field(..., description="Custom collection ID")
    product_id:    int           = Field(..., description="Product ID to add")
    position:      Optional[int] = Field(default=None, description="Manual sort position (1-based)")


@mcp.tool(
    name="shopify_add_product_to_collection",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_add_product_to_collection(params: AddProductToCollectionInput) -> str:
    """Add a product to a custom (manual) collection. For smart collections, tag the product instead."""
    try:
        collect: Dict[str, Any] = {"collection_id": params.collection_id, "product_id": params.product_id}
        if params.position:
            collect["position"] = params.position
        data = await _request("POST", "collects.json", body={"collect": collect})
        return _fmt(data.get("collect", data))
    except Exception as e:
        return _error(e)


class RemoveProductFromCollectionInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collect_id: int = Field(..., description="The collect ID linking product<>collection (from list_collects or add response)")


@mcp.tool(
    name="shopify_remove_product_from_collection",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_remove_product_from_collection(params: RemoveProductFromCollectionInput) -> str:
    """Remove a product from a custom collection by deleting its collect record."""
    try:
        await _request("DELETE", f"collects/{params.collect_id}.json")
        return _fmt({"deleted": True, "collect_id": params.collect_id})
    except Exception as e:
        return _error(e)


class ListCollectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    collection_id: Optional[int] = Field(default=None, description="Filter by collection")
    product_id:    Optional[int] = Field(default=None, description="Filter by product")
    limit:         Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_collects",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_collects(params: ListCollectsInput) -> str:
    """List collect records (product↔collection links). Use to find a collect_id for removal."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.collection_id:
            p["collection_id"] = params.collection_id
        if params.product_id:
            p["product_id"] = params.product_id
        data = await _request("GET", "collects.json", params=p)
        collects = data.get("collects", [])
        return _fmt({"count": len(collects), "collects": collects})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# INVENTORY
# ═══════════════════════════════════════════════════════════════════════════

class ListInventoryLocationsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_list_locations",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_locations(params: ListInventoryLocationsInput) -> str:
    """List all inventory locations for the store."""
    try:
        data      = await _request("GET", "locations.json")
        locations = data.get("locations", [])
        return _fmt({"count": len(locations), "locations": locations})
    except Exception as e:
        return _error(e)


class GetInventoryLevelsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    location_id:         Optional[int] = Field(default=None, description="Filter by location ID")
    inventory_item_ids:  Optional[str] = Field(default=None, description="Comma-separated inventory item IDs")


@mcp.tool(
    name="shopify_get_inventory_levels",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_inventory_levels(params: GetInventoryLevelsInput) -> str:
    """Get inventory levels for specific locations or inventory items."""
    try:
        p: Dict[str, Any] = {}
        if params.location_id:
            p["location_ids"] = params.location_id
        if params.inventory_item_ids:
            p["inventory_item_ids"] = params.inventory_item_ids
        data   = await _request("GET", "inventory_levels.json", params=p)
        levels = data.get("inventory_levels", [])
        return _fmt({"count": len(levels), "inventory_levels": levels})
    except Exception as e:
        return _error(e)


class SetInventoryLevelInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    inventory_item_id: int = Field(..., description="Inventory item ID")
    location_id:       int = Field(..., description="Location ID")
    available:         int = Field(..., description="Available quantity to set")


@mcp.tool(
    name="shopify_set_inventory_level",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_inventory_level(params: SetInventoryLevelInput) -> str:
    """Set the available inventory for an item at a location."""
    try:
        body = {
            "inventory_item_id": params.inventory_item_id,
            "location_id":       params.location_id,
            "available":         params.available,
        }
        data = await _request("POST", "inventory_levels/set.json", body=body)
        return _fmt(data.get("inventory_level", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# FULFILLMENTS
# ═══════════════════════════════════════════════════════════════════════════

class ListFulfillmentsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id: int           = Field(..., description="Order ID")
    limit:    Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_fulfillments",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_fulfillments(params: ListFulfillmentsInput) -> str:
    """List fulfillments for a specific order."""
    try:
        p            = {"limit": params.limit}
        data         = await _request("GET", f"orders/{params.order_id}/fulfillments.json", params=p)
        fulfillments = data.get("fulfillments", [])
        return _fmt({"count": len(fulfillments), "fulfillments": fulfillments})
    except Exception as e:
        return _error(e)


class CreateFulfillmentInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    order_id:         int                        = Field(..., description="Order ID to fulfill")
    location_id:      int                        = Field(..., description="Location ID fulfilling from")
    tracking_number:  Optional[str]              = Field(default=None)
    tracking_company: Optional[str]              = Field(default=None, description="e.g. UPS, FedEx, USPS")
    tracking_url:     Optional[str]              = Field(default=None)
    line_items:       Optional[List[Dict[str, Any]]] = Field(default=None, description="Specific line items (omit for all)")
    notify_customer:  Optional[bool]             = Field(default=True, description="Send shipping notification email")


@mcp.tool(
    name="shopify_create_fulfillment",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_fulfillment(params: CreateFulfillmentInput) -> str:
    """Create a fulfillment for an order (ship items)."""
    try:
        fulfillment: Dict[str, Any] = {"location_id": params.location_id}
        for field in ["tracking_number", "tracking_company", "tracking_url", "line_items", "notify_customer"]:
            val = getattr(params, field)
            if val is not None:
                fulfillment[field] = val
        data = await _request(
            "POST",
            f"orders/{params.order_id}/fulfillments.json",
            body={"fulfillment": fulfillment},
        )
        return _fmt(data.get("fulfillment", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# SHOP INFO
# ═══════════════════════════════════════════════════════════════════════════

class EmptyInput(BaseModel):
    model_config = ConfigDict(extra="forbid")


@mcp.tool(
    name="shopify_get_shop",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_shop(params: EmptyInput) -> str:
    """Get store information: name, domain, plan, currency, timezone, etc."""
    try:
        data = await _request("GET", "shop.json")
        return _fmt(data.get("shop", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# WEBHOOKS
# ═══════════════════════════════════════════════════════════════════════════

class ListWebhooksInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit: Optional[int] = Field(default=50, ge=1, le=250)
    topic: Optional[str] = Field(default=None, description="Filter by topic, e.g. orders/create")


@mcp.tool(
    name="shopify_list_webhooks",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_webhooks(params: ListWebhooksInput) -> str:
    """List configured webhooks."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        if params.topic:
            p["topic"] = params.topic
        data     = await _request("GET", "webhooks.json", params=p)
        webhooks = data.get("webhooks", [])
        return _fmt({"count": len(webhooks), "webhooks": webhooks})
    except Exception as e:
        return _error(e)


class CreateWebhookInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    topic:   str           = Field(..., description="Webhook topic, e.g. orders/create, products/update")
    address: str           = Field(..., description="URL to receive the webhook POST")
    format:  Optional[str] = Field(default="json", description="json or xml")


@mcp.tool(
    name="shopify_create_webhook",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_webhook(params: CreateWebhookInput) -> str:
    """Create a new webhook subscription."""
    try:
        webhook = {"topic": params.topic, "address": params.address, "format": params.format}
        data    = await _request("POST", "webhooks.json", body={"webhook": webhook})
        return _fmt(data.get("webhook", data))
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# PAGES (About, Shipping, Returns, etc. — requires read_content/write_content scopes)
# ═══════════════════════════════════════════════════════════════════════════

class ListPagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:       Optional[int] = Field(default=50, ge=1, le=250)
    since_id:    Optional[int] = Field(default=None)
    title:       Optional[str] = Field(default=None, description="Filter by exact title")
    handle:      Optional[str] = Field(default=None, description="Filter by URL handle")
    fields:      Optional[str] = Field(default=None, description="Comma-separated fields to include")


@mcp.tool(
    name="shopify_list_pages",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_pages(params: ListPagesInput) -> str:
    """List pages on the store (About, Shipping, Returns, etc.)."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["since_id", "title", "handle", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data  = await _request("GET", "pages.json", params=p)
        pages = data.get("pages", [])
        return _fmt({"count": len(pages), "pages": pages})
    except Exception as e:
        return _error(e)


class GetPageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: int = Field(..., description="Page ID")


@mcp.tool(
    name="shopify_get_page",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_page(params: GetPageInput) -> str:
    """Retrieve a single page by ID, including its body_html."""
    try:
        data = await _request("GET", f"pages/{params.page_id}.json")
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)


class CreatePageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:      str                          = Field(..., min_length=1)
    body_html:  Optional[str]                = Field(default=None, description="HTML content")
    author:     Optional[str]                = Field(default=None)
    handle:     Optional[str]                = Field(default=None, description="URL handle. Auto-generated from title if omitted")
    published:  Optional[bool]               = Field(default=False, description="Publish immediately. Default False (draft)")
    metafields: Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline metafields (e.g. SEO title/description)")


@mcp.tool(
    name="shopify_create_page",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_page(params: CreatePageInput) -> str:
    """Create a new page (About, Shipping, Returns, etc.)."""
    try:
        page: Dict[str, Any] = {"title": params.title, "published": bool(params.published)}
        for field in ["body_html", "author", "handle"]:
            val = getattr(params, field)
            if val is not None:
                page[field] = val
        if params.metafields:
            page["metafields"] = params.metafields
        data = await _request("POST", "pages.json", body={"page": page})
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)


class UpdatePageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    page_id:   int           = Field(..., description="Page ID")
    title:     Optional[str] = Field(default=None)
    body_html: Optional[str] = Field(default=None)
    author:    Optional[str] = Field(default=None)
    handle:    Optional[str] = Field(default=None)
    published: Optional[bool] = Field(default=None)


@mcp.tool(
    name="shopify_update_page",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_page(params: UpdatePageInput) -> str:
    """Update an existing page. Only provided fields are changed."""
    try:
        payload: Dict[str, Any] = {"id": params.page_id}
        for field in ["title", "body_html", "author", "handle"]:
            val = getattr(params, field)
            if val is not None:
                payload[field] = val
        if params.published is not None:
            payload["published"] = params.published
        data = await _request("PUT", f"pages/{params.page_id}.json", body={"page": payload})
        return _fmt(data.get("page", data))
    except Exception as e:
        return _error(e)


class DeletePageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    page_id: int = Field(..., description="Page ID")


@mcp.tool(
    name="shopify_delete_page",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_page(params: DeletePageInput) -> str:
    """Permanently delete a page. This cannot be undone."""
    try:
        await _request("DELETE", f"pages/{params.page_id}.json")
        return _fmt({"deleted": True, "page_id": params.page_id})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# METAFIELDS (SEO title/description, size charts, custom data on any resource)
# ═══════════════════════════════════════════════════════════════════════════
#
# Shopify stores SEO title and meta description as metafields:
#   namespace = "global", key = "title_tag"        (single_line_text_field)
#   namespace = "global", key = "description_tag"  (multi_line_text_field)
#
# Use shopify_set_seo for a simple SEO title/description workflow, or
# shopify_set_metafield for any other metafield.

_METAFIELD_OWNER_PATHS = {
    "product":    "products",
    "collection": "collections",
    "page":       "pages",
    "blog":       "blogs",
    "article":    "articles",
    "variant":    "variants",
    "customer":   "customers",
    "order":      "orders",
    "shop":       None,  # Shop-level metafields use /metafields.json directly
}


def _metafield_endpoint(owner_resource: str, owner_id: Optional[int], metafield_id: Optional[int] = None) -> str:
    if owner_resource not in _METAFIELD_OWNER_PATHS:
        raise ValueError(f"Unknown owner_resource '{owner_resource}'. Must be one of: {', '.join(_METAFIELD_OWNER_PATHS)}")
    base = _METAFIELD_OWNER_PATHS[owner_resource]
    if owner_resource == "shop":
        return f"metafields/{metafield_id}.json" if metafield_id else "metafields.json"
    if owner_id is None:
        raise ValueError(f"owner_id is required for owner_resource='{owner_resource}'")
    if metafield_id:
        return f"{base}/{owner_id}/metafields/{metafield_id}.json"
    return f"{base}/{owner_id}/metafields.json"


class ListMetafieldsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_resource: str           = Field(..., description="product, collection, page, blog, article, variant, customer, order, or shop")
    owner_id:       Optional[int] = Field(default=None, description="Required except when owner_resource='shop'")
    namespace:      Optional[str] = Field(default=None, description="Filter by namespace, e.g. 'global' for SEO fields")
    key:            Optional[str] = Field(default=None, description="Filter by key, e.g. 'title_tag'")
    limit:          Optional[int] = Field(default=50, ge=1, le=250)


@mcp.tool(
    name="shopify_list_metafields",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_metafields(params: ListMetafieldsInput) -> str:
    """List metafields on a resource. Useful to check existing SEO title/description before updating."""
    try:
        endpoint = _metafield_endpoint(params.owner_resource, params.owner_id)
        p: Dict[str, Any] = {"limit": params.limit}
        if params.namespace:
            p["namespace"] = params.namespace
        if params.key:
            p["key"] = params.key
        data       = await _request("GET", endpoint, params=p)
        metafields = data.get("metafields", [])
        return _fmt({"count": len(metafields), "metafields": metafields})
    except Exception as e:
        return _error(e)


class SetMetafieldInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    owner_resource: str           = Field(..., description="product, collection, page, blog, article, variant, customer, order, or shop")
    owner_id:       Optional[int] = Field(default=None, description="Required except when owner_resource='shop'")
    namespace:      str           = Field(..., description="Metafield namespace, e.g. 'global' for SEO, 'custom' for custom fields")
    key:            str           = Field(..., description="Metafield key, e.g. 'title_tag', 'description_tag', 'size_chart'")
    value:          str           = Field(..., description="The value to set (strings only — for numbers/JSON, pass as a string)")
    type:           Optional[str] = Field(default="single_line_text_field", description="single_line_text_field, multi_line_text_field, number_integer, boolean, json, url, etc.")


@mcp.tool(
    name="shopify_set_metafield",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_metafield(params: SetMetafieldInput) -> str:
    """Create or update a metafield on a product, collection, page, etc. Upserts by (namespace, key)."""
    try:
        endpoint = _metafield_endpoint(params.owner_resource, params.owner_id)
        metafield = {
            "namespace": params.namespace,
            "key":       params.key,
            "value":     params.value,
            "type":      params.type,
        }
        # Shopify upserts on POST when (namespace, key) already exists on the owner
        data = await _request("POST", endpoint, body={"metafield": metafield})
        return _fmt(data.get("metafield", data))
    except Exception as e:
        return _error(e)


class DeleteMetafieldInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    owner_resource: str           = Field(..., description="product, collection, page, blog, article, variant, customer, order, or shop")
    owner_id:       Optional[int] = Field(default=None, description="Required except when owner_resource='shop'")
    metafield_id:   int           = Field(..., description="The metafield ID (from list_metafields)")


@mcp.tool(
    name="shopify_delete_metafield",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_metafield(params: DeleteMetafieldInput) -> str:
    """Permanently delete a metafield by ID."""
    try:
        endpoint = _metafield_endpoint(params.owner_resource, params.owner_id, params.metafield_id)
        await _request("DELETE", endpoint)
        return _fmt({"deleted": True, "metafield_id": params.metafield_id})
    except Exception as e:
        return _error(e)


class SetSeoInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    owner_resource:   str           = Field(..., description="product, collection, page, or article")
    owner_id:         int           = Field(..., description="Product/collection/page/article ID")
    seo_title:        Optional[str] = Field(default=None, description="SEO title tag (~50-60 characters recommended)")
    seo_description:  Optional[str] = Field(default=None, description="Meta description (~120-155 characters recommended)")


@mcp.tool(
    name="shopify_set_seo",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_set_seo(params: SetSeoInput) -> str:
    """Set SEO title and/or meta description on a product, collection, page, or article. Convenience wrapper around set_metafield."""
    if params.owner_resource not in ("product", "collection", "page", "article"):
        return _error(ValueError(f"owner_resource must be 'product', 'collection', 'page', or 'article' (got '{params.owner_resource}')"))
    if params.seo_title is None and params.seo_description is None:
        return _error(ValueError("Provide at least one of seo_title or seo_description"))

    results: Dict[str, Any] = {}
    try:
        endpoint = _metafield_endpoint(params.owner_resource, params.owner_id)
        if params.seo_title is not None:
            body = {"metafield": {
                "namespace": "global",
                "key":       "title_tag",
                "value":     params.seo_title,
                "type":      "single_line_text_field",
            }}
            data = await _request("POST", endpoint, body=body)
            results["title_tag"] = data.get("metafield", data)
        if params.seo_description is not None:
            body = {"metafield": {
                "namespace": "global",
                "key":       "description_tag",
                "value":     params.seo_description,
                "type":      "multi_line_text_field",
            }}
            data = await _request("POST", endpoint, body=body)
            results["description_tag"] = data.get("metafield", data)
        return _fmt(results)
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# BLOGS (containers — a store can have multiple blogs, e.g. "News", "Style Guide")
# ═══════════════════════════════════════════════════════════════════════════

class ListBlogsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:    Optional[int] = Field(default=50, ge=1, le=250)
    since_id: Optional[int] = Field(default=None)
    handle:   Optional[str] = Field(default=None, description="Filter by handle")
    fields:   Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_blogs",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_blogs(params: ListBlogsInput) -> str:
    """List blogs on the store. A blog is a container that holds articles."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["since_id", "handle", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data  = await _request("GET", "blogs.json", params=p)
        blogs = data.get("blogs", [])
        return _fmt({"count": len(blogs), "blogs": blogs})
    except Exception as e:
        return _error(e)


class GetBlogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id: int = Field(..., description="Blog ID")


@mcp.tool(
    name="shopify_get_blog",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_blog(params: GetBlogInput) -> str:
    """Retrieve a single blog by ID."""
    try:
        data = await _request("GET", f"blogs/{params.blog_id}.json")
        return _fmt(data.get("blog", data))
    except Exception as e:
        return _error(e)


class CreateBlogInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    title:          str                          = Field(..., min_length=1)
    handle:         Optional[str]                = Field(default=None, description="URL handle. Auto-generated from title")
    commentable:    Optional[str]                = Field(default="no", description="no, moderate, or yes")
    feedburner:     Optional[str]                = Field(default=None, description="FeedBurner URL")
    feedburner_url: Optional[str]                = Field(default=None)
    tags:           Optional[str]                = Field(default=None, description="Comma-separated tags (used across articles)")
    template_suffix: Optional[str]               = Field(default=None, description="Alternate theme template suffix")
    metafields:     Optional[List[Dict[str, Any]]] = Field(default=None)


@mcp.tool(
    name="shopify_create_blog",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_blog(params: CreateBlogInput) -> str:
    """Create a new blog (a container for articles)."""
    try:
        blog: Dict[str, Any] = {"title": params.title}
        for field in ["handle", "commentable", "feedburner", "feedburner_url", "tags", "template_suffix"]:
            val = getattr(params, field)
            if val is not None:
                blog[field] = val
        if params.metafields:
            blog["metafields"] = params.metafields
        data = await _request("POST", "blogs.json", body={"blog": blog})
        return _fmt(data.get("blog", data))
    except Exception as e:
        return _error(e)


class UpdateBlogInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    blog_id:         int           = Field(..., description="Blog ID")
    title:           Optional[str] = Field(default=None)
    handle:          Optional[str] = Field(default=None)
    commentable:     Optional[str] = Field(default=None)
    tags:            Optional[str] = Field(default=None)
    template_suffix: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_blog",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_blog(params: UpdateBlogInput) -> str:
    """Update an existing blog's settings."""
    try:
        payload: Dict[str, Any] = {"id": params.blog_id}
        for field in ["title", "handle", "commentable", "tags", "template_suffix"]:
            val = getattr(params, field)
            if val is not None:
                payload[field] = val
        data = await _request("PUT", f"blogs/{params.blog_id}.json", body={"blog": payload})
        return _fmt(data.get("blog", data))
    except Exception as e:
        return _error(e)


class DeleteBlogInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id: int = Field(..., description="Blog ID")


@mcp.tool(
    name="shopify_delete_blog",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_blog(params: DeleteBlogInput) -> str:
    """Permanently delete a blog and ALL its articles. This cannot be undone."""
    try:
        await _request("DELETE", f"blogs/{params.blog_id}.json")
        return _fmt({"deleted": True, "blog_id": params.blog_id})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# ARTICLES (blog posts)
# ═══════════════════════════════════════════════════════════════════════════

class ListArticlesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id:          Optional[int] = Field(default=None, description="Scope to one blog. Omit to list all articles")
    limit:            Optional[int] = Field(default=50, ge=1, le=250)
    since_id:         Optional[int] = Field(default=None)
    handle:           Optional[str] = Field(default=None)
    tag:              Optional[str] = Field(default=None, description="Filter by tag")
    author:           Optional[str] = Field(default=None)
    published_status: Optional[str] = Field(default=None, description="published, unpublished, or any")
    fields:           Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_articles",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_articles(params: ListArticlesInput) -> str:
    """List articles, optionally scoped to a specific blog."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["since_id", "handle", "tag", "author", "published_status", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        endpoint = f"blogs/{params.blog_id}/articles.json" if params.blog_id else "articles.json"
        data     = await _request("GET", endpoint, params=p)
        articles = data.get("articles", [])
        return _fmt({"count": len(articles), "articles": articles})
    except Exception as e:
        return _error(e)


class GetArticleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id:    int = Field(..., description="Parent blog ID")
    article_id: int = Field(..., description="Article ID")


@mcp.tool(
    name="shopify_get_article",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_article(params: GetArticleInput) -> str:
    """Retrieve a single article by ID, including its body_html."""
    try:
        data = await _request("GET", f"blogs/{params.blog_id}/articles/{params.article_id}.json")
        return _fmt(data.get("article", data))
    except Exception as e:
        return _error(e)


class CreateArticleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    blog_id:         int                          = Field(..., description="Parent blog ID (use list_blogs to find it)")
    title:           str                          = Field(..., min_length=1)
    body_html:       Optional[str]                = Field(default=None, description="HTML content of the post")
    author:          Optional[str]                = Field(default=None, description="Author name")
    summary_html:    Optional[str]                = Field(default=None, description="HTML excerpt shown in blog listing")
    tags:            Optional[str]                = Field(default=None, description="Comma-separated tags")
    handle:          Optional[str]                = Field(default=None)
    published:       Optional[bool]               = Field(default=False, description="Publish immediately. Default False (draft)")
    published_at:    Optional[str]                = Field(default=None, description="ISO 8601 datetime to schedule future publishing")
    image_src:       Optional[str]                = Field(default=None, description="Featured image URL")
    image_alt:       Optional[str]                = Field(default=None, description="Featured image alt text (SEO)")
    template_suffix: Optional[str]                = Field(default=None)
    metafields:      Optional[List[Dict[str, Any]]] = Field(default=None, description="Inline metafields (e.g. SEO title/description)")


@mcp.tool(
    name="shopify_create_article",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_article(params: CreateArticleInput) -> str:
    """Create a new blog article. Use list_blogs first to find the blog_id."""
    try:
        article: Dict[str, Any] = {"title": params.title, "published": bool(params.published)}
        for field in ["body_html", "author", "summary_html", "tags", "handle", "published_at", "template_suffix"]:
            val = getattr(params, field)
            if val is not None:
                article[field] = val
        if params.image_src:
            img: Dict[str, Any] = {"src": params.image_src}
            if params.image_alt:
                img["alt"] = params.image_alt
            article["image"] = img
        if params.metafields:
            article["metafields"] = params.metafields
        data = await _request("POST", f"blogs/{params.blog_id}/articles.json", body={"article": article})
        return _fmt(data.get("article", data))
    except Exception as e:
        return _error(e)


class UpdateArticleInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    blog_id:         int           = Field(..., description="Parent blog ID")
    article_id:      int           = Field(..., description="Article ID")
    title:           Optional[str] = Field(default=None)
    body_html:       Optional[str] = Field(default=None)
    author:          Optional[str] = Field(default=None)
    summary_html:    Optional[str] = Field(default=None)
    tags:            Optional[str] = Field(default=None)
    handle:          Optional[str] = Field(default=None)
    published:       Optional[bool] = Field(default=None)
    published_at:    Optional[str] = Field(default=None)
    image_src:       Optional[str] = Field(default=None)
    image_alt:       Optional[str] = Field(default=None)
    template_suffix: Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_article",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_article(params: UpdateArticleInput) -> str:
    """Update an existing article. Only provided fields are changed."""
    try:
        payload: Dict[str, Any] = {"id": params.article_id}
        for field in ["title", "body_html", "author", "summary_html", "tags", "handle", "published_at", "template_suffix"]:
            val = getattr(params, field)
            if val is not None:
                payload[field] = val
        if params.published is not None:
            payload["published"] = params.published
        if params.image_src:
            img: Dict[str, Any] = {"src": params.image_src}
            if params.image_alt:
                img["alt"] = params.image_alt
            payload["image"] = img
        data = await _request("PUT", f"blogs/{params.blog_id}/articles/{params.article_id}.json", body={"article": payload})
        return _fmt(data.get("article", data))
    except Exception as e:
        return _error(e)


class DeleteArticleInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    blog_id:    int = Field(..., description="Parent blog ID")
    article_id: int = Field(..., description="Article ID")


@mcp.tool(
    name="shopify_delete_article",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_article(params: DeleteArticleInput) -> str:
    """Permanently delete an article. This cannot be undone."""
    try:
        await _request("DELETE", f"blogs/{params.blog_id}/articles/{params.article_id}.json")
        return _fmt({"deleted": True, "article_id": params.article_id})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# REDIRECTS (URL redirects — critical for SEO when renaming collections/pages)
# ═══════════════════════════════════════════════════════════════════════════

class ListRedirectsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    limit:    Optional[int] = Field(default=50, ge=1, le=250)
    since_id: Optional[int] = Field(default=None)
    path:     Optional[str] = Field(default=None, description="Filter by source path, e.g. /old-url")
    target:   Optional[str] = Field(default=None, description="Filter by destination URL")
    fields:   Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_list_redirects",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_redirects(params: ListRedirectsInput) -> str:
    """List URL redirects on the store."""
    try:
        p: Dict[str, Any] = {"limit": params.limit}
        for field in ["since_id", "path", "target", "fields"]:
            val = getattr(params, field)
            if val is not None:
                p[field] = val
        data      = await _request("GET", "redirects.json", params=p)
        redirects = data.get("redirects", [])
        return _fmt({"count": len(redirects), "redirects": redirects})
    except Exception as e:
        return _error(e)


class GetRedirectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    redirect_id: int = Field(..., description="Redirect ID")


@mcp.tool(
    name="shopify_get_redirect",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_redirect(params: GetRedirectInput) -> str:
    """Retrieve a single redirect by ID."""
    try:
        data = await _request("GET", f"redirects/{params.redirect_id}.json")
        return _fmt(data.get("redirect", data))
    except Exception as e:
        return _error(e)


class CreateRedirectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    path:   str = Field(..., description="Source path on your store, e.g. /collections/old-name")
    target: str = Field(..., description="Destination: relative path (/collections/new-name) or full URL")


@mcp.tool(
    name="shopify_create_redirect",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_create_redirect(params: CreateRedirectInput) -> str:
    """Create a URL redirect. Use when renaming collection/page/product handles to preserve SEO."""
    try:
        redirect = {"path": params.path, "target": params.target}
        data = await _request("POST", "redirects.json", body={"redirect": redirect})
        return _fmt(data.get("redirect", data))
    except Exception as e:
        return _error(e)


class UpdateRedirectInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    redirect_id: int           = Field(..., description="Redirect ID")
    path:        Optional[str] = Field(default=None)
    target:      Optional[str] = Field(default=None)


@mcp.tool(
    name="shopify_update_redirect",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_redirect(params: UpdateRedirectInput) -> str:
    """Update an existing redirect."""
    try:
        payload: Dict[str, Any] = {"id": params.redirect_id}
        if params.path is not None:
            payload["path"] = params.path
        if params.target is not None:
            payload["target"] = params.target
        data = await _request("PUT", f"redirects/{params.redirect_id}.json", body={"redirect": payload})
        return _fmt(data.get("redirect", data))
    except Exception as e:
        return _error(e)


class DeleteRedirectInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    redirect_id: int = Field(..., description="Redirect ID")


@mcp.tool(
    name="shopify_delete_redirect",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_redirect(params: DeleteRedirectInput) -> str:
    """Permanently delete a redirect."""
    try:
        await _request("DELETE", f"redirects/{params.redirect_id}.json")
        return _fmt({"deleted": True, "redirect_id": params.redirect_id})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# PRODUCT IMAGES — attach from src, rename, alt text, reorder, delete
# ═══════════════════════════════════════════════════════════════════════════

class ListProductImagesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID")


@mcp.tool(
    name="shopify_list_product_images",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_product_images(params: ListProductImagesInput) -> str:
    """List all images on a product, with their IDs, positions, alt text, and src URLs."""
    try:
        data   = await _request("GET", f"products/{params.product_id}/images.json")
        images = data.get("images", [])
        return _fmt({"count": len(images), "images": images})
    except Exception as e:
        return _error(e)


class AddProductImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:  int                 = Field(..., description="Product ID to attach the image to")
    src:         str                 = Field(..., description="Public image URL (e.g. a Shopify CDN src)")
    alt:         Optional[str]       = Field(default=None, description="Alt text (accessibility + image SEO)")
    position:    Optional[int]       = Field(default=None, ge=1, description="1-based display order")
    variant_ids: Optional[List[int]] = Field(default=None, description="Attach image to specific variant IDs")
    filename:    Optional[str]       = Field(default=None, description="Desired filename (e.g. 'blue-dress-front.jpg')")


@mcp.tool(
    name="shopify_add_product_image",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False, "openWorldHint": True},
)
async def shopify_add_product_image(params: AddProductImageInput) -> str:
    """Attach an image to a product from a public URL (src). Supports alt text, position, variant mapping, and a custom filename."""
    try:
        image: Dict[str, Any] = {"src": params.src}
        for field in ["alt", "position", "variant_ids", "filename"]:
            val = getattr(params, field)
            if val is not None:
                image[field] = val
        data = await _request(
            "POST", f"products/{params.product_id}/images.json",
            body={"image": image},
        )
        return _fmt(data.get("image", data))
    except Exception as e:
        return _error(e)


class UpdateProductImageInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    product_id:  int                 = Field(..., description="Product ID")
    image_id:    int                 = Field(..., description="Image ID")
    alt:         Optional[str]       = Field(default=None, description="Alt text (SEO + accessibility)")
    position:    Optional[int]       = Field(default=None, ge=1, description="1-based display order")
    filename:    Optional[str]       = Field(default=None, description="Rename the image file (e.g. 'blue-dress-side.jpg')")
    variant_ids: Optional[List[int]] = Field(default=None, description="Replace the set of variant IDs this image is attached to")


@mcp.tool(
    name="shopify_update_product_image",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_product_image(params: UpdateProductImageInput) -> str:
    """Update an existing product image: alt text, filename (rename), position, or variant mapping."""
    try:
        image: Dict[str, Any] = {"id": params.image_id}
        for field in ["alt", "position", "filename", "variant_ids"]:
            val = getattr(params, field)
            if val is not None:
                image[field] = val
        data = await _request(
            "PUT", f"products/{params.product_id}/images/{params.image_id}.json",
            body={"image": image},
        )
        return _fmt(data.get("image", data))
    except Exception as e:
        return _error(e)


class DeleteProductImageInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    product_id: int = Field(..., description="Product ID")
    image_id:   int = Field(..., description="Image ID to delete")


@mcp.tool(
    name="shopify_delete_product_image",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_product_image(params: DeleteProductImageInput) -> str:
    """Delete a single image from a product."""
    try:
        await _request("DELETE", f"products/{params.product_id}/images/{params.image_id}.json")
        return _fmt({"deleted": True, "product_id": params.product_id, "image_id": params.image_id})
    except Exception as e:
        return _error(e)


# ═══════════════════════════════════════════════════════════════════════════
# THEMES & THEME ASSETS — read/write theme files
# ═══════════════════════════════════════════════════════════════════════════
# The live theme has role="main". Updating assets on the main theme takes
# effect on the storefront immediately — for safer edits, duplicate the
# theme in admin first and edit the unpublished copy, then publish it.

class ListThemesInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    role: Optional[str] = Field(default=None, description="Filter by role: main, unpublished, demo, development")


@mcp.tool(
    name="shopify_list_themes",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_themes(params: ListThemesInput) -> str:
    """List all themes installed on the store. The live theme has role='main'."""
    try:
        data   = await _request("GET", "themes.json")
        themes = data.get("themes", [])
        if params.role:
            themes = [t for t in themes if t.get("role") == params.role]
        return _fmt({"count": len(themes), "themes": themes})
    except Exception as e:
        return _error(e)


class ListThemeAssetsInput(BaseModel):
    model_config = ConfigDict(extra="forbid")
    theme_id: int           = Field(..., description="Theme ID")
    fields:   Optional[str] = Field(default=None, description="Comma-separated fields, e.g. 'key,content_type,size,updated_at'")


@mcp.tool(
    name="shopify_list_theme_assets",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_list_theme_assets(params: ListThemeAssetsInput) -> str:
    """List all assets (files) in a theme. Returns metadata only — use shopify_get_theme_asset to read file contents."""
    try:
        p: Dict[str, Any] = {}
        if params.fields:
            p["fields"] = params.fields
        data   = await _request("GET", f"themes/{params.theme_id}/assets.json", params=p)
        assets = data.get("assets", [])
        return _fmt({"count": len(assets), "assets": assets})
    except Exception as e:
        return _error(e)


class GetThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    key:      str = Field(..., min_length=1, description="Asset path, e.g. 'templates/product.json' or 'sections/header.liquid'")


@mcp.tool(
    name="shopify_get_theme_asset",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_get_theme_asset(params: GetThemeAssetInput) -> str:
    """Fetch the contents of a single theme asset (liquid template, JSON config, CSS, JS, etc.)."""
    try:
        data = await _request(
            "GET", f"themes/{params.theme_id}/assets.json",
            params={"asset[key]": params.key},
        )
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)


class UpdateThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id:   int           = Field(..., description="Theme ID")
    key:        str           = Field(..., min_length=1, description="Asset path, e.g. 'sections/header.liquid'")
    value:      Optional[str] = Field(default=None, description="Text content of the asset (liquid, JSON, CSS, JS)")
    attachment: Optional[str] = Field(default=None, description="Base64-encoded content (for binary assets like images)")
    src:        Optional[str] = Field(default=None, description="Public URL to copy the asset from")
    source_key: Optional[str] = Field(default=None, description="Key of another asset in the same theme to copy from")


@mcp.tool(
    name="shopify_update_theme_asset",
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_update_theme_asset(params: UpdateThemeAssetInput) -> str:
    """Create or update a theme asset. Provide exactly one of: value, attachment, src, or source_key.

    WARNING: If theme_id is the live theme (role='main'), changes are published instantly.
    Safer workflow: duplicate the theme in Shopify admin first, edit the copy, publish when ready.
    """
    try:
        asset: Dict[str, Any] = {"key": params.key}
        for field in ["value", "attachment", "src", "source_key"]:
            val = getattr(params, field)
            if val is not None:
                asset[field] = val
        if len(asset) == 1:
            return "Provide at least one of: value, attachment, src, or source_key."
        data = await _request(
            "PUT", f"themes/{params.theme_id}/assets.json",
            body={"asset": asset},
        )
        return _fmt(data.get("asset", data))
    except Exception as e:
        return _error(e)


class DeleteThemeAssetInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    theme_id: int = Field(..., description="Theme ID")
    key:      str = Field(..., min_length=1, description="Asset path to delete")


@mcp.tool(
    name="shopify_delete_theme_asset",
    annotations={"readOnlyHint": False, "destructiveHint": True, "idempotentHint": True, "openWorldHint": True},
)
async def shopify_delete_theme_asset(params: DeleteThemeAssetInput) -> str:
    """Delete a single asset from a theme. Be careful on the live theme."""
    try:
        await _request(
            "DELETE", f"themes/{params.theme_id}/assets.json",
            params={"asset[key]": params.key},
        )
        return _fmt({"deleted": True, "theme_id": params.theme_id, "key": params.key})
    except Exception as e:
        return _error(e)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    mcp.run(transport=MCP_TRANSPORT)
