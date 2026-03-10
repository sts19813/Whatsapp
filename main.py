import json
import os
import re
from datetime import datetime
from typing import Any
from zoneinfo import ZoneInfo

import requests
from fastapi import FastAPI, Request

from services.ai_service import analizar_intencion_taquilla
from services.audio_service import transcribir_audio
from services.chatbot_admin_service import ChatbotAdminClient, ChatbotAdminError, is_uuid
from services.db_service import (
    get_memories,
    get_memory_value,
    get_recent_messages,
    get_user,
    init_db,
    save_message,
    upsert_memory,
    upsert_user,
)
from services.image_service import procesar_imagen


app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN", "mi_webhook_verificacion_123")
WHATSAPP_TOKEN = os.getenv("WHATSAPP_TOKEN")
PHONE_NUMBER_ID = os.getenv("PHONE_NUMBER_ID")
BOT_NAME = "Kiro AI Taquilla Admin"

ADMIN_API = ChatbotAdminClient()

VALID_SCOPE_EVENTS = {"all", "upcoming", "past", "today"}
VALID_SCOPE_AVAILABILITY = {"all", "upcoming", "today"}
VALID_SALE_TYPE = {"all", "ticket", "registration"}
JSON_BLOCK_RE = re.compile(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", re.IGNORECASE)


@app.on_event("startup")
def on_startup() -> None:
    init_db()


def clamp(value: Any, min_value: int, max_value: int, default: int) -> int:
    try:
        parsed = int(value)
    except Exception:
        parsed = default
    return max(min_value, min(max_value, parsed))


def normalize_text(value: str) -> str:
    return " ".join((value or "").strip().lower().split())


def compact(value: Any, max_len: int = 220) -> str:
    text = " ".join(str(value or "").split())
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def safe_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def safe_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def pick(data: dict[str, Any], *keys: str, default: Any = "") -> Any:
    if not isinstance(data, dict):
        return default

    for key in keys:
        value = data.get(key)
        if value not in (None, ""):
            return value

    for nested_key in ("event", "sale", "registration"):
        nested = data.get(nested_key)
        if not isinstance(nested, dict):
            continue
        for key in keys:
            value = nested.get(key)
            if value not in (None, ""):
                return value

    return default


def format_money(value: Any) -> str:
    if value in (None, ""):
        return "N/D"
    try:
        amount = float(value)
    except Exception:
        return str(value)
    if amount.is_integer():
        return f"${int(amount):,}"
    return f"${amount:,.2f}"


def extract_json_payload(text: str) -> dict[str, Any] | None:
    content = (text or "").strip()
    if not content:
        return None

    candidates: list[str] = []
    block_match = JSON_BLOCK_RE.search(content)
    if block_match:
        candidates.append(block_match.group(1))

    if content.startswith("{") and content.endswith("}"):
        candidates.append(content)

    first = content.find("{")
    last = content.rfind("}")
    if first >= 0 and last > first:
        candidates.append(content[first : last + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            continue
    return None


def contexto_fecha_actual() -> str:
    now = datetime.now(ZoneInfo("America/Mexico_City"))
    dias = ["lunes", "martes", "miercoles", "jueves", "viernes", "sabado", "domingo"]
    meses = [
        "enero",
        "febrero",
        "marzo",
        "abril",
        "mayo",
        "junio",
        "julio",
        "agosto",
        "septiembre",
        "octubre",
        "noviembre",
        "diciembre",
    ]
    return (
        f"Fecha local: {dias[now.weekday()]} {now.day} de {meses[now.month - 1]} de {now.year}. "
        f"Hora aprox: {now.strftime('%H:%M')} (America/Mexico_City)."
    )


def contexto_compacto(sender: str) -> str:
    user = get_user(sender) or {}
    memories = get_memories(sender, limit=10)
    recent = get_recent_messages(sender, limit=6)

    memory_lines = [
        f"- {m.get('key', '')}: {compact(m.get('value', ''), 120)}"
        for m in memories
        if (m.get("key") or "").strip()
    ]
    history_lines = [
        f"{row.get('role', 'user')}: {compact(row.get('content', ''), 140)}"
        for row in recent[-4:]
    ]

    return (
        f"{contexto_fecha_actual()}\n"
        f"display_name: {user.get('display_name', 'sin nombre')}\n"
        f"memoria:\n{chr(10).join(memory_lines) if memory_lines else '- (sin memoria)'}\n"
        f"historial:\n{chr(10).join(history_lines) if history_lines else '(sin historial)'}"
    )


def remember_event(sender: str, event_id: str, event_name: str = "") -> None:
    if not is_uuid(event_id):
        return
    upsert_memory(sender, "last_event_id", event_id, 5)
    if event_name:
        upsert_memory(sender, "last_event_name", event_name, 4)


def extract_event_identity(item: dict[str, Any]) -> tuple[str, str]:
    event_id = str(pick(item, "event_id", "id", "uuid", "event_uuid", default="")).strip()
    event_name = str(pick(item, "event_name", "name", "title", default="")).strip()
    return event_id, event_name


def format_event_option(index: int, item: dict[str, Any]) -> str:
    event_id, event_name = extract_event_identity(item)
    event_date = pick(item, "start_at", "starts_at", "start_date", "date", "event_date")

    parts = [f"{index}. {event_name or 'Evento'}"]
    if event_date:
        parts.append(f"fecha: {event_date}")
    if event_id:
        parts.append(f"id: {event_id}")
    return " | ".join(parts)


def resolve_event_id(
    sender: str,
    intent_data: dict[str, Any],
    required: bool = True,
) -> tuple[str, str]:
    direct_event_id = str(intent_data.get("event_id") or "").strip()
    if is_uuid(direct_event_id):
        return direct_event_id, ""

    event_query = str(intent_data.get("event_query") or intent_data.get("q") or "").strip()
    if event_query:
        payload = ADMIN_API.get_events(q=event_query, scope="all", limit=5)
        rows = safe_list(payload.get("data"))

        if not rows:
            return "", f"No encontre eventos para: {event_query}"

        if len(rows) == 1:
            event_id, event_name = extract_event_identity(safe_dict(rows[0]))
            if is_uuid(event_id):
                remember_event(sender, event_id, event_name)
                return event_id, ""

        options = [format_event_option(i, safe_dict(row)) for i, row in enumerate(rows[:5], start=1)]
        return "", "Encontre varios eventos. Indica el UUID exacto:\n" + "\n".join(options)

    remembered = str(get_memory_value(sender, "last_event_id") or "").strip()
    if is_uuid(remembered):
        return remembered, ""

    if required:
        return "", "Necesito el event_id del evento. Primero puedes pedir: proximos eventos."
    return "", ""


def respuesta_ayuda() -> str:
    return (
        "Asistente de taquilla listo.\n\n"
        "Consultas soportadas:\n"
        "1) Eventos: resumen, proximos y detalle.\n"
        "2) Disponibilidad: tickets, registros y items vendibles.\n"
        "3) Ventas: overview, busqueda y ultimas ventas.\n"
        "4) PDF: busqueda para reimpresion/reenvio.\n"
        "5) Venta en efectivo: crea venta por /taquilla/sell-cash.\n\n"
        "Ejemplos:\n"
        "- Dame proximos eventos\n"
        "- Disponibilidad del evento 0f33... (UUID)\n"
        "- Buscar ventas por email cliente@dominio.com hoy\n"
        "- Reenviar PDF de referencia ABC123\n"
        "- Vender en efectivo con este JSON {event_id, buyer_name, buyer_email, cart}"
    )


def handle_events(sender: str, intent_data: dict[str, Any]) -> str:
    q = str(intent_data.get("q") or intent_data.get("event_query") or "").strip()
    scope = str(intent_data.get("scope") or "all").strip().lower()
    if scope not in VALID_SCOPE_EVENTS:
        scope = "all"
    limit = clamp(intent_data.get("limit"), 1, 200, 10)

    payload = ADMIN_API.get_events(q=q, scope=scope, limit=limit)
    rows = safe_list(payload.get("data"))
    if not rows:
        return "No se encontraron eventos con ese filtro."

    first_id, first_name = extract_event_identity(safe_dict(rows[0]))
    remember_event(sender, first_id, first_name)

    meta = safe_dict(payload.get("meta"))
    total = pick(meta, "total", "count", "total_items", default="")
    showing = min(len(rows), 8)
    lines = [f"Eventos encontrados: {showing}{f' de {total}' if total else ''}"]
    for i, row in enumerate(rows[:8], start=1):
        item = safe_dict(row)
        event_id, event_name = extract_event_identity(item)
        event_date = pick(item, "start_at", "starts_at", "start_date", "date", "event_date")
        available = pick(item, "tickets_available", "available_tickets", "stock_available")
        sold = pick(item, "tickets_sold", "sold_tickets", "sales_count", "sold")
        revenue = pick(item, "total_sales_amount", "sales_amount", "revenue", "total_amount")

        pieces = [f"{i}. {event_name or 'Evento'}"]
        if event_date:
            pieces.append(f"fecha: {event_date}")
        if available not in ("", None):
            pieces.append(f"disp: {available}")
        if sold not in ("", None):
            pieces.append(f"vendidos: {sold}")
        if revenue not in ("", None):
            pieces.append(f"ventas: {format_money(revenue)}")
        if event_id:
            pieces.append(f"id: {event_id}")
        lines.append(" | ".join(pieces))

    return "\n".join(lines)


def handle_events_upcoming(sender: str, intent_data: dict[str, Any]) -> str:
    q = str(intent_data.get("q") or intent_data.get("event_query") or "").strip()
    limit = clamp(intent_data.get("limit"), 1, 200, 10)

    payload = ADMIN_API.get_events_upcoming(q=q, limit=limit)
    rows = safe_list(payload.get("data"))
    if not rows:
        return "No hay proximos eventos para ese filtro."

    first_id, first_name = extract_event_identity(safe_dict(rows[0]))
    remember_event(sender, first_id, first_name)

    lines = [f"Proximos eventos ({min(len(rows), 8)}):"]
    for i, row in enumerate(rows[:8], start=1):
        item = safe_dict(row)
        event_id, event_name = extract_event_identity(item)
        event_date = pick(item, "start_at", "starts_at", "start_date", "date", "event_date")
        pieces = [f"{i}. {event_name or 'Evento'}"]
        if event_date:
            pieces.append(f"fecha: {event_date}")
        if event_id:
            pieces.append(f"id: {event_id}")
        lines.append(" | ".join(pieces))
    return "\n".join(lines)


def handle_event_detail(sender: str, intent_data: dict[str, Any]) -> str:
    event_id, issue = resolve_event_id(sender, intent_data, required=True)
    if issue:
        return issue

    payload = ADMIN_API.get_event_detail(event_id)
    event = safe_dict(payload.get("event"))
    ticket_breakdown = safe_list(payload.get("ticket_breakdown"))
    registration = safe_dict(payload.get("registration_breakdown"))

    event_name = str(pick(event, "name", "event_name", "title", default="")).strip() or event_id
    remember_event(sender, event_id, event_name)

    lines = [f"Detalle evento: {event_name}", f"id: {event_id}"]

    event_date = pick(event, "start_at", "starts_at", "start_date", "date")
    venue = pick(event, "venue", "venue_name", "location")
    status = pick(event, "status", "event_status")
    if event_date:
        lines.append(f"fecha: {event_date}")
    if venue:
        lines.append(f"sede: {venue}")
    if status:
        lines.append(f"estatus: {status}")

    if ticket_breakdown:
        lines.append("Tickets:")
        for item in ticket_breakdown[:8]:
            row = safe_dict(item)
            ticket_id = pick(row, "id", "ticket_id", "uuid", default="")
            ticket_name = pick(row, "name", "ticket_name", "label", "concept", default="Ticket")
            price = pick(row, "unit_price", "price", default="")
            stock = pick(row, "stock_available", "available", "remaining", default="")
            sold = pick(row, "sold", "sold_count", "sold_qty", default="")
            can_sell = pick(row, "can_sell_cash", default="")

            parts = [f"- {ticket_name}"]
            if ticket_id:
                parts.append(f"id: {ticket_id}")
            if price not in ("", None):
                parts.append(f"precio: {format_money(price)}")
            if stock not in ("", None):
                parts.append(f"disp: {stock}")
            if sold not in ("", None):
                parts.append(f"vendidos: {sold}")
            if can_sell not in ("", None):
                parts.append(f"cash: {'si' if bool(can_sell) else 'no'}")
            lines.append(" | ".join(parts))
    else:
        lines.append("Tickets: sin desglose disponible.")

    if registration:
        reg_enabled = pick(registration, "enabled", default="")
        reg_price = pick(registration, "unit_price", "price", default="")
        reg_slots = pick(registration, "slots_available", "available_slots", "remaining", default="")
        reg_can_sell = pick(registration, "can_sell_cash", default="")
        reg_parts = ["Registro:"]
        if reg_enabled not in ("", None):
            reg_parts.append(f"habilitado: {'si' if bool(reg_enabled) else 'no'}")
        if reg_price not in ("", None):
            reg_parts.append(f"precio: {format_money(reg_price)}")
        if reg_slots not in ("", None):
            reg_parts.append(f"cupos: {reg_slots}")
        if reg_can_sell not in ("", None):
            reg_parts.append(f"cash: {'si' if bool(reg_can_sell) else 'no'}")
        lines.append(" | ".join(reg_parts))
    else:
        lines.append("Registro: sin desglose disponible.")

    return "\n".join(lines)


def handle_availability(sender: str, intent_data: dict[str, Any]) -> str:
    requested_event = str(intent_data.get("event_id") or "").strip()
    event_id = requested_event if is_uuid(requested_event) else ""
    q = str(intent_data.get("q") or intent_data.get("event_query") or "").strip()
    scope = str(intent_data.get("scope") or "upcoming").strip().lower()
    if scope not in VALID_SCOPE_AVAILABILITY:
        scope = "upcoming"
    limit = clamp(intent_data.get("limit"), 1, 200, 10)

    if not event_id and not q:
        remembered = str(get_memory_value(sender, "last_event_id") or "").strip()
        if is_uuid(remembered):
            event_id = remembered

    payload = ADMIN_API.get_availability(event_id=event_id, q=q, scope=scope, limit=limit)
    rows = safe_list(payload.get("data"))
    if not rows:
        return "No hay disponibilidad para ese filtro."

    first_id, first_name = extract_event_identity(safe_dict(rows[0]))
    remember_event(sender, first_id, first_name)

    lines = [f"Disponibilidad ({min(len(rows), 5)} eventos):"]
    for i, raw in enumerate(rows[:5], start=1):
        row = safe_dict(raw)
        current_event_id, current_event_name = extract_event_identity(row)
        title = f"{i}. {current_event_name or 'Evento'}"
        if current_event_id:
            title += f" | id: {current_event_id}"
        lines.append(title)

        tickets = safe_list(row.get("tickets"))
        if tickets:
            lines.append("Tickets:")
            for ticket_raw in tickets[:8]:
                ticket = safe_dict(ticket_raw)
                ticket_id = pick(ticket, "id", "ticket_id", "uuid", default="")
                ticket_name = pick(ticket, "name", "label", "ticket_name", default="Ticket")
                unit_price = pick(ticket, "unit_price", "price", default="")
                stock = pick(ticket, "stock_available", "available", "remaining", default="")
                can_sell_cash = pick(ticket, "can_sell_cash", default="")

                parts = [f"- {ticket_name}"]
                if ticket_id:
                    parts.append(f"id: {ticket_id}")
                if unit_price not in ("", None):
                    parts.append(f"precio: {format_money(unit_price)}")
                if stock not in ("", None):
                    parts.append(f"disp: {stock}")
                if can_sell_cash not in ("", None):
                    parts.append(f"cash: {'si' if bool(can_sell_cash) else 'no'}")
                lines.append(" | ".join(parts))
        else:
            lines.append("Tickets: sin items.")

        registration = safe_dict(row.get("registration"))
        if registration:
            enabled = pick(registration, "enabled", default="")
            price = pick(registration, "unit_price", "price", default="")
            slots = pick(registration, "slots_available", "available_slots", "remaining", default="")
            can_sell_cash = pick(registration, "can_sell_cash", default="")

            reg_parts = ["Registro:"]
            if enabled not in ("", None):
                reg_parts.append(f"habilitado: {'si' if bool(enabled) else 'no'}")
            if price not in ("", None):
                reg_parts.append(f"precio: {format_money(price)}")
            if slots not in ("", None):
                reg_parts.append(f"cupos: {slots}")
            if can_sell_cash not in ("", None):
                reg_parts.append(f"cash: {'si' if bool(can_sell_cash) else 'no'}")
            lines.append(" | ".join(reg_parts))

        sellable_items = safe_list(row.get("sellable_items"))
        if sellable_items:
            render = []
            for sellable_raw in sellable_items[:8]:
                sellable = safe_dict(sellable_raw)
                item_type = str(pick(sellable, "type", default="item"))
                item_id = str(pick(sellable, "id", "ticket_id", default="")).strip()
                item_name = str(pick(sellable, "name", "label", "concept", default="")).strip()
                label = f"{item_type}:{item_id}" if item_id else item_type
                if item_name:
                    label = f"{label} ({item_name})"
                render.append(label)
            lines.append("Items vendibles: " + "; ".join(render))

    lines.append("Tip: usa los ids de tickets para ejecutar venta en efectivo.")
    return "\n".join(lines)


def handle_sales_overview(sender: str, intent_data: dict[str, Any]) -> str:
    requested_event = str(intent_data.get("event_id") or "").strip()
    event_id = requested_event if is_uuid(requested_event) else ""
    if not event_id and (intent_data.get("event_query") or intent_data.get("q")):
        event_id, issue = resolve_event_id(sender, intent_data, required=False)
        if issue:
            return issue

    date_exact = str(intent_data.get("date") or "").strip()
    date_from = str(intent_data.get("from") or "").strip()
    date_to = str(intent_data.get("to") or "").strip()
    if date_exact and not date_from and not date_to:
        date_from = date_exact
        date_to = date_exact

    payload = ADMIN_API.get_sales_overview(event_id=event_id, date_from=date_from, date_to=date_to)
    overview = safe_dict(payload.get("data")) if isinstance(payload.get("data"), dict) else safe_dict(payload)
    if not overview:
        return "No se pudo construir el overview de ventas."

    lines = ["Resumen global de ventas:"]
    if event_id:
        lines.append(f"event_id: {event_id}")
    if date_from or date_to:
        lines.append(f"rango: {date_from or 'inicio'} -> {date_to or 'fin'}")

    used = set()
    known_fields = [
        ("total_amount", "Monto total"),
        ("total_count", "Total operaciones"),
        ("total_items", "Total items"),
        ("ticket_amount", "Monto tickets"),
        ("ticket_count", "Tickets vendidos"),
        ("registration_amount", "Monto registros"),
        ("registration_count", "Registros"),
    ]
    for key, label in known_fields:
        value = overview.get(key)
        if value in (None, ""):
            continue
        used.add(key)
        if "amount" in key or "monto" in key:
            lines.append(f"- {label}: {format_money(value)}")
        else:
            lines.append(f"- {label}: {value}")

    for key, value in overview.items():
        if key in used:
            continue
        if isinstance(value, (dict, list)):
            continue
        if value in (None, ""):
            continue
        label = key.replace("_", " ")
        if any(token in key for token in ("amount", "price", "revenue", "total")):
            lines.append(f"- {label}: {format_money(value)}")
        else:
            lines.append(f"- {label}: {value}")

    return "\n".join(lines)


def _resolve_event_for_sales(sender: str, intent_data: dict[str, Any]) -> tuple[str, str]:
    requested_event = str(intent_data.get("event_id") or "").strip()
    if is_uuid(requested_event):
        return requested_event, ""
    if intent_data.get("event_query") or intent_data.get("q"):
        return resolve_event_id(sender, intent_data, required=False)

    remembered = str(get_memory_value(sender, "last_event_id") or "").strip()
    if is_uuid(remembered):
        return remembered, ""
    return "", ""


def handle_sales_latest(sender: str, intent_data: dict[str, Any]) -> str:
    event_id, issue = _resolve_event_for_sales(sender, intent_data)
    if issue:
        return issue

    limit = clamp(intent_data.get("limit"), 1, 50, 10)
    payload = ADMIN_API.get_sales_latest(event_id=event_id, limit=limit)
    rows = safe_list(payload.get("data"))
    if not rows:
        return "No hay ventas recientes con ese filtro."

    lines = [f"Ultimas ventas ({min(len(rows), 8)}):"]
    for i, raw in enumerate(rows[:8], start=1):
        row = safe_dict(raw)
        customer_name = pick(row, "customer_name", "name", default="Cliente")
        email = pick(row, "customer_email", "email", default="")
        price = pick(row, "price", "amount", default="")
        sold_at = pick(row, "sold_at", "created_at", "date", default="")
        reference = pick(row, "reference", default="")
        pdf_url = pick(row, "pdf_url", "reprint_pdf_url", default="")

        lines.append(f"{i}. {customer_name} | {format_money(price)} | {sold_at or 'sin fecha'}")
        if email:
            lines.append(f"   email: {email}")
        if reference:
            lines.append(f"   referencia: {reference}")
        if pdf_url:
            lines.append(f"   pdf: {pdf_url}")

    return "\n".join(lines)


def handle_sales_search(sender: str, intent_data: dict[str, Any], pdf_only: bool = False) -> str:
    sale_type = str(intent_data.get("sale_type") or "all").strip().lower()
    if sale_type not in VALID_SALE_TYPE:
        sale_type = "all"

    q = str(intent_data.get("q") or "").strip()
    reference = str(intent_data.get("reference") or "").strip()
    if reference and not q:
        q = reference

    name = str(intent_data.get("name") or "").strip()
    email = str(intent_data.get("email") or "").strip()
    date_exact = str(intent_data.get("date") or "").strip()
    date_from = str(intent_data.get("from") or "").strip()
    date_to = str(intent_data.get("to") or "").strip()
    if date_exact and (date_from or date_to):
        date_exact = ""

    event_id, issue = _resolve_event_for_sales(sender, intent_data)
    if issue:
        return issue

    limit = clamp(intent_data.get("limit"), 1, 200, 20)
    payload = ADMIN_API.search_sales(
        sale_type=sale_type,
        q=q,
        name=name,
        email=email,
        event_id=event_id,
        exact_date=date_exact,
        date_from=date_from,
        date_to=date_to,
        limit=limit,
    )

    rows = safe_list(payload.get("data"))
    if pdf_only:
        rows = [row for row in rows if pick(safe_dict(row), "pdf_url", "reprint_pdf_url", default="")]

    if not rows:
        if pdf_only:
            return "No encontre ventas con PDF para esos filtros."
        return "No encontre ventas con esos filtros."

    lines = [f"Resultados ventas ({min(len(rows), 10)}):"]
    for i, raw in enumerate(rows[:10], start=1):
        row = safe_dict(raw)
        sale_mode = pick(row, "sale_type", "type", default="sale")
        customer_name = pick(row, "customer_name", "name", default="Cliente")
        concept = pick(row, "concept", default="concepto")
        price = pick(row, "price", "amount", default="")
        sold_at = pick(row, "sold_at", "created_at", default="")
        reference_value = pick(row, "reference", default="")
        pdf_url = pick(row, "pdf_url", "reprint_pdf_url", default="")
        customer_email = pick(row, "customer_email", "email", default="")
        event_name = pick(row, "event_name", "name", default="")
        event_id = str(pick(row, "event_id", default="")).strip()

        if is_uuid(event_id):
            remember_event(sender, event_id, str(event_name))

        lines.append(
            f"{i}. {sale_mode} | {customer_name} | {concept} | {format_money(price)} | {sold_at or 'sin fecha'}"
        )
        if reference_value:
            lines.append(f"   referencia: {reference_value}")
        if customer_email:
            lines.append(f"   email: {customer_email}")
        if event_name:
            lines.append(f"   evento: {event_name}")
        if pdf_url:
            lines.append(f"   pdf: {pdf_url}")

    if pdf_only:
        lines.append("Usa el campo pdf para reenviar por WhatsApp.")
    return "\n".join(lines)


def normalize_cart_items(raw_cart: Any) -> tuple[list[dict[str, Any]], str]:
    items = []
    for raw_item in safe_list(raw_cart):
        item = safe_dict(raw_item)
        item_type = str(item.get("type") or "").strip().lower()
        qty = clamp(item.get("qty"), 1, 9999, 1)

        if item_type == "ticket":
            item_id = str(item.get("id") or "").strip()
            if not item_id:
                return [], "Cada item de tipo ticket requiere id."
            items.append({"type": "ticket", "id": item_id, "qty": qty})
            continue

        if item_type == "registration":
            normalized = {"type": "registration", "qty": qty}
            if item.get("price") not in (None, ""):
                try:
                    normalized["price"] = float(item.get("price"))
                except Exception:
                    return [], "El campo price en registration debe ser numerico."
            items.append(normalized)
            continue

        return [], "Tipo de item invalido en cart. Usa ticket o registration."

    if not items:
        return [], "Cart vacio. Incluye al menos un item ticket o registration."
    return items, ""


def build_sell_payload(
    sender: str,
    text: str,
    intent_data: dict[str, Any],
) -> tuple[dict[str, Any] | None, str]:
    parsed_payload = extract_json_payload(text) or {}

    requested_event = str(parsed_payload.get("event_id") or intent_data.get("event_id") or "").strip()
    event_id = requested_event if is_uuid(requested_event) else ""
    if not event_id:
        if intent_data.get("event_query") or intent_data.get("q"):
            event_id, issue = resolve_event_id(sender, intent_data, required=True)
            if issue:
                return None, issue
        else:
            remembered = str(get_memory_value(sender, "last_event_id") or "").strip()
            if is_uuid(remembered):
                event_id = remembered
    if not event_id:
        return None, "Necesito event_id valido para vender en efectivo."

    buyer_name = str(parsed_payload.get("buyer_name") or intent_data.get("buyer_name") or "").strip()
    buyer_email = str(parsed_payload.get("buyer_email") or intent_data.get("buyer_email") or "").strip()
    buyer_phone = str(parsed_payload.get("buyer_phone") or intent_data.get("buyer_phone") or "").strip()

    if not buyer_name:
        return None, "Falta buyer_name para crear la venta."
    if not buyer_email:
        return None, "Falta buyer_email para crear la venta."

    cart_input = parsed_payload.get("cart")
    if not isinstance(cart_input, list):
        cart_input = intent_data.get("cart")
    cart, cart_error = normalize_cart_items(cart_input)
    if cart_error:
        return None, cart_error

    payload: dict[str, Any] = {
        "event_id": event_id,
        "buyer_name": buyer_name,
        "buyer_email": buyer_email,
        "cart": cart,
    }
    if buyer_phone:
        payload["buyer_phone"] = buyer_phone
    if isinstance(parsed_payload.get("registration_form"), dict):
        payload["registration_form"] = parsed_payload["registration_form"]

    return payload, ""


def handle_sell_cash(sender: str, text: str, intent_data: dict[str, Any]) -> str:
    payload, issue = build_sell_payload(sender, text, intent_data)
    if issue:
        return issue
    assert payload is not None

    response = ADMIN_API.sell_cash(payload)
    sale = safe_dict(response.get("sale"))
    items = safe_dict(response.get("items"))

    reference = pick(sale, "reference", default="")
    total_amount = pick(sale, "total_amount", "amount", "total", default="")
    total_items = pick(sale, "total_items", "items_count", default="")
    reprint_pdf = pick(sale, "reprint_pdf_url", "pdf_url", default="")

    if reference:
        upsert_memory(sender, "last_sale_reference", str(reference), 4)
    if reprint_pdf:
        upsert_memory(sender, "last_sale_pdf_url", str(reprint_pdf), 4)

    lines = ["Venta en efectivo creada."]
    if reference:
        lines.append(f"referencia: {reference}")
    if total_amount not in ("", None):
        lines.append(f"total: {format_money(total_amount)}")
    if total_items not in ("", None):
        lines.append(f"items: {total_items}")
    if reprint_pdf:
        lines.append(f"pdf reimpresion: {reprint_pdf}")

    ticket_items = safe_list(items.get("tickets"))
    registration_items = safe_list(items.get("registrations"))
    if ticket_items or registration_items:
        lines.append("Desglose:")
        for raw in ticket_items[:10]:
            row = safe_dict(raw)
            name = pick(row, "name", "ticket_name", "concept", default="ticket")
            qty = pick(row, "qty", "quantity", default="")
            price = pick(row, "unit_price", "price", default="")
            detail = f"- ticket {name}"
            if qty not in ("", None):
                detail += f" x{qty}"
            if price not in ("", None):
                detail += f" ({format_money(price)})"
            lines.append(detail)
        for raw in registration_items[:10]:
            row = safe_dict(raw)
            qty = pick(row, "qty", "quantity", default="")
            price = pick(row, "unit_price", "price", default="")
            detail = "- registration"
            if qty not in ("", None):
                detail += f" x{qty}"
            if price not in ("", None):
                detail += f" ({format_money(price)})"
            lines.append(detail)

    return "\n".join(lines)


def infer_intent_fallback(text: str) -> str:
    t = normalize_text(text)

    if any(word in t for word in ("ayuda", "comandos", "que puedes", "como funciona")):
        return "help"
    if "proximo" in t and "evento" in t:
        return "events_upcoming"
    if "detalle" in t and "evento" in t:
        return "event_detail"
    if any(word in t for word in ("disponibilidad", "boleto disponible", "stock", "taquilla")):
        return "availability"
    if "overview" in t or ("resumen" in t and "venta" in t):
        return "sales_overview"
    if ("ultimas" in t or "ultimos" in t or "recientes" in t) and "venta" in t:
        return "sales_latest"
    if any(word in t for word in ("pdf", "reimpresion", "reenviar", "reenvio")):
        return "pdf_lookup"
    if "buscar" in t and "venta" in t:
        return "sales_search"
    if any(word in t for word in ("vender", "venta en efectivo", "sell-cash")):
        return "sell_cash"
    if "evento" in t:
        return "events"
    return "unknown"


def procesar_texto_taquilla(sender: str, text: str, message_type: str = "text") -> str:
    contenido = (text or "").strip()
    if not contenido:
        return "No recibi texto. Intenta de nuevo."

    first_turn = len(get_recent_messages(sender, limit=1)) == 0
    save_message(sender, "user", contenido, message_type)

    if first_turn and normalize_text(contenido) in {"hola", "buenas", "hello", "hi"}:
        return (
            "Hola. Soy tu asistente de taquilla para admin.\n\n"
            "Puedo consultar eventos, disponibilidad, ventas y links PDF de reimpresion.\n"
            "Escribe ayuda para ver ejemplos."
        )

    if not ADMIN_API.base_url:
        return "Config faltante: CHATBOT_ADMIN_BASE_URL."
    if not ADMIN_API.token:
        return "Config faltante: CHATBOT_ADMIN_TOKEN."

    intent_data: dict[str, Any]
    try:
        intent_data = analizar_intencion_taquilla(contexto_compacto(sender), contenido)
    except Exception:
        intent_data = {}

    if not isinstance(intent_data, dict):
        intent_data = {}

    intent = str(intent_data.get("intent") or "").strip().lower()
    if not intent or intent == "unknown":
        intent = infer_intent_fallback(contenido)
        intent_data["intent"] = intent

    needs_clarification = bool(intent_data.get("needs_clarification"))
    clarification_question = str(intent_data.get("clarification_question") or "").strip()
    if needs_clarification and clarification_question:
        return clarification_question

    try:
        if intent == "help":
            return respuesta_ayuda()
        if intent == "events":
            return handle_events(sender, intent_data)
        if intent == "events_upcoming":
            return handle_events_upcoming(sender, intent_data)
        if intent == "event_detail":
            return handle_event_detail(sender, intent_data)
        if intent == "availability":
            return handle_availability(sender, intent_data)
        if intent == "sales_overview":
            return handle_sales_overview(sender, intent_data)
        if intent == "sales_search":
            return handle_sales_search(sender, intent_data, pdf_only=False)
        if intent == "sales_latest":
            return handle_sales_latest(sender, intent_data)
        if intent == "pdf_lookup":
            return handle_sales_search(sender, intent_data, pdf_only=True)
        if intent == "sell_cash":
            return handle_sell_cash(sender, contenido, intent_data)

        return (
            "No pude identificar la accion exacta.\n"
            "Puedes pedir ayuda o usar comandos como: proximos eventos, disponibilidad, ventas, pdf o vender."
        )
    except ChatbotAdminError as api_error:
        return api_error.message
    except Exception as exc:
        print("ERROR EN PROCESO TAQUILLA:", exc)
        return "Hubo un error interno procesando la solicitud."


def obtener_display_name(value: dict[str, Any]) -> str | None:
    contacts = value.get("contacts") or []
    if not contacts:
        return None
    profile = contacts[0].get("profile") or {}
    return profile.get("name")


def descargar_media(media_id: str) -> bytes | None:
    try:
        url = f"https://graph.facebook.com/v22.0/{media_id}"
        headers = {"Authorization": f"Bearer {WHATSAPP_TOKEN}"}

        info = requests.get(url, headers=headers, timeout=20)
        media_info = info.json() if info.ok else {}
        media_url = media_info.get("url")
        if not media_url:
            return None

        media_response = requests.get(media_url, headers=headers, timeout=30)
        return media_response.content if media_response.ok else None
    except Exception as exc:
        print("ERROR DESCARGANDO MEDIA:", exc)
        return None


def enviar_mensaje_whatsapp(numero: str, mensaje: str) -> None:
    try:
        url = f"https://graph.facebook.com/v22.0/{PHONE_NUMBER_ID}/messages"
        headers = {
            "Authorization": f"Bearer {WHATSAPP_TOKEN}",
            "Content-Type": "application/json",
        }
        payload = {
            "messaging_product": "whatsapp",
            "to": numero,
            "type": "text",
            "text": {"body": mensaje[:4000]},
        }
        requests.post(url, headers=headers, json=payload, timeout=20)
    except Exception as exc:
        print("ERROR ENVIANDO MENSAJE:", exc)


@app.get("/")
def health() -> dict[str, Any]:
    return {
        "status": "running",
        "bot": BOT_NAME,
        "admin_api_configured": ADMIN_API.configured,
        "admin_api_base_url": ADMIN_API.base_url,
        "admin_auth_mode": ADMIN_API.auth_mode,
        "admin_token_len": len(ADMIN_API.token or ""),
    }


@app.get("/webhook")
async def verify_webhook(request: Request):
    mode = request.query_params.get("hub.mode")
    token = request.query_params.get("hub.verify_token")
    challenge = request.query_params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:
        return int(challenge)
    return {"error": "Verification failed"}


@app.post("/webhook")
async def receive_message(request: Request):
    try:
        data = await request.json()

        for entry in data.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})
                if "messages" not in value:
                    continue

                display_name = obtener_display_name(value)

                for message in value.get("messages", []):
                    sender = message.get("from")
                    msg_type = message.get("type")
                    if not sender or not msg_type:
                        continue

                    upsert_user(sender, display_name=display_name)
                    respuesta_final = None

                    if msg_type == "text":
                        text = message.get("text", {}).get("body", "")
                        respuesta_final = procesar_texto_taquilla(sender, text, "text")

                    elif msg_type == "audio":
                        media_id = message.get("audio", {}).get("id")
                        audio_bytes = descargar_media(media_id) if media_id else None
                        if audio_bytes:
                            texto_audio = transcribir_audio(audio_bytes)
                            respuesta_final = procesar_texto_taquilla(sender, texto_audio, "audio")
                        else:
                            respuesta_final = "No pude descargar el audio. Intenta de nuevo."

                    elif msg_type == "image":
                        media_id = message.get("image", {}).get("id")
                        image_bytes = descargar_media(media_id) if media_id else None
                        if image_bytes:
                            descripcion = procesar_imagen(image_bytes)
                            respuesta_final = procesar_texto_taquilla(sender, descripcion, "image")
                        else:
                            respuesta_final = "No pude descargar la imagen. Intenta de nuevo."

                    elif msg_type == "document":
                        respuesta_final = (
                            "Para reimpresion PDF usa busqueda de ventas, por ejemplo:\n"
                            "- Buscar venta referencia ABC123\n"
                            "- Buscar venta email cliente@dominio.com"
                        )

                    else:
                        respuesta_final = (
                            "Tipo de mensaje no soportado aun. Envia texto, audio o imagen."
                        )

                    if respuesta_final:
                        save_message(sender, "assistant", respuesta_final, "text")
                        enviar_mensaje_whatsapp(sender, respuesta_final)

        return {"status": "ok"}
    except Exception as exc:
        print("ERROR:", exc)
        return {"error": str(exc)}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
