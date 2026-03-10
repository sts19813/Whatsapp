import os
import re
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse

import requests


UUID_RE = re.compile(
    r"^[0-9a-fA-F]{8}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{4}-"
    r"[0-9a-fA-F]{12}$"
)


def is_uuid(value: str) -> bool:
    return bool(UUID_RE.match((value or "").strip()))


def _clean_query(params: dict[str, Any]) -> dict[str, Any]:
    clean: dict[str, Any] = {}
    for key, value in params.items():
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        clean[key] = value
    return clean


def _clean_env(value: Optional[str]) -> str:
    raw = (value or "").strip()
    if len(raw) >= 2 and ((raw[0] == raw[-1] == '"') or (raw[0] == raw[-1] == "'")):
        return raw[1:-1].strip()
    return raw


def _normalize_base_url(value: str) -> str:
    raw = _clean_env(value).rstrip("/")
    if not raw:
        return ""

    parsed = urlparse(raw)
    # Si viene sin ruta (solo dominio), usa la ruta default de la Admin API.
    if parsed.scheme and parsed.netloc and (not parsed.path or parsed.path == "/"):
        parsed = parsed._replace(path="/api/chatbot/admin")
        return urlunparse(parsed).rstrip("/")

    # Si ya trae una ruta parcial de chatbot/admin, normaliza al prefijo esperado.
    path = parsed.path.rstrip("/")
    if path in {"/api/chatbot", "/api/chatbot/"}:
        parsed = parsed._replace(path="/api/chatbot/admin")
        return urlunparse(parsed).rstrip("/")

    return raw


class ChatbotAdminError(Exception):
    def __init__(self, status_code: int, message: str, payload: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.payload = payload


class ChatbotAdminClient:
    def __init__(
        self,
        base_url: Optional[str] = None,
        token: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
    ):
        self.base_url = _normalize_base_url(base_url or os.getenv("CHATBOT_ADMIN_BASE_URL", ""))
        self.token = _clean_env(token or os.getenv("CHATBOT_ADMIN_TOKEN", ""))
        self.timeout_seconds = int(timeout_seconds or os.getenv("CHATBOT_ADMIN_TIMEOUT", "25"))
        self.auth_mode = _clean_env(os.getenv("CHATBOT_ADMIN_AUTH_MODE", "x_token")).lower()

    @property
    def configured(self) -> bool:
        return bool(self.base_url and self.token)

    def _headers(self, mode: Optional[str] = None) -> dict[str, str]:
        auth_mode = (mode or self.auth_mode or "x_token").lower()
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json",
        }
        if auth_mode in {"x_token", "", "default"}:
            headers["X-Chatbot-Token"] = self.token
        elif auth_mode == "bearer":
            headers["Authorization"] = f"Bearer {self.token}"
        elif auth_mode == "both":
            headers["X-Chatbot-Token"] = self.token
            headers["Authorization"] = f"Bearer {self.token}"
        else:
            headers["X-Chatbot-Token"] = self.token
        return headers

    def _message_from_error(self, status_code: int, payload: Any) -> str:
        if status_code == 401:
            return "Token admin invalido. Verifica CHATBOT_ADMIN_TOKEN."
        if status_code == 503:
            return "El servidor indica que el token admin no esta configurado (503)."
        if status_code == 422:
            return "Solicitud invalida. Revisa event_id, fechas y formato del body."

        if isinstance(payload, dict):
            detail = payload.get("detail") or payload.get("message") or payload.get("error")
            if isinstance(detail, str) and detail.strip():
                return detail.strip()
        return f"Error al consumir Admin API ({status_code})."

    def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict[str, Any]] = None,
        json_body: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        if not self.base_url:
            raise ChatbotAdminError(
                status_code=500,
                message="Falta configurar CHATBOT_ADMIN_BASE_URL.",
            )
        if not self.token:
            raise ChatbotAdminError(
                status_code=503,
                message="Falta configurar CHATBOT_ADMIN_TOKEN en el bot.",
            )

        url = f"{self.base_url}{path}"
        query = _clean_query(params or {})
        primary_mode = self.auth_mode if self.auth_mode in {"x_token", "bearer", "both"} else "x_token"
        retry_modes: list[str] = [primary_mode]
        for fallback_mode in ("x_token", "bearer", "both"):
            if fallback_mode not in retry_modes:
                retry_modes.append(fallback_mode)

        last_response = None
        last_payload: Any = None
        last_mode = primary_mode

        for mode in retry_modes:
            try:
                response = requests.request(
                    method=method.upper(),
                    url=url,
                    headers=self._headers(mode=mode),
                    params=query,
                    json=json_body,
                    timeout=self.timeout_seconds,
                )
            except requests.RequestException as exc:
                raise ChatbotAdminError(
                    status_code=502,
                    message=f"No se pudo conectar con Admin API: {exc}",
                ) from exc

            try:
                payload: Any = response.json()
            except ValueError:
                payload = {"raw": (response.text or "").strip()}

            if response.status_code != 401:
                last_response = response
                last_payload = payload
                last_mode = mode
                break

            last_response = response
            last_payload = payload
            last_mode = mode

        response = last_response
        payload = last_payload
        if response is None:
            raise ChatbotAdminError(
                status_code=502,
                message="No se obtuvo respuesta de Admin API.",
            )

        if response.status_code >= 400:
            if response.status_code == 401:
                token_len = len(self.token)
                mode_txt = last_mode or primary_mode
                diagnostic = (
                    f"Token admin invalido. Verifica CHATBOT_ADMIN_TOKEN. "
                    f"(base_url={self.base_url}, auth_mode={mode_txt}, token_len={token_len})"
                )
                raise ChatbotAdminError(
                    status_code=response.status_code,
                    message=diagnostic,
                    payload=payload,
                )
            raise ChatbotAdminError(
                status_code=response.status_code,
                message=self._message_from_error(response.status_code, payload),
                payload=payload,
            )

        if isinstance(payload, dict):
            return payload
        return {"data": payload}

    def get_events(
        self,
        q: str = "",
        scope: str = "",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/events",
            params={"q": q, "scope": scope, "limit": limit},
        )

    def get_events_upcoming(
        self,
        q: str = "",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        try:
            return self._request(
                "GET",
                "/events/upcoming",
                params={"q": q, "limit": limit},
            )
        except ChatbotAdminError as exc:
            # Compatibilidad con backends que usan /events?scope=upcoming
            # en vez de /events/upcoming.
            if exc.status_code in {404, 405}:
                return self.get_events(q=q, scope="upcoming", limit=limit)
            raise

    def get_event_detail(self, event_id: str) -> dict[str, Any]:
        return self._request("GET", f"/events/{event_id}")

    def get_sales_overview(
        self,
        event_id: str = "",
        date_from: str = "",
        date_to: str = "",
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/sales/overview",
            params={"event_id": event_id, "from": date_from, "to": date_to},
        )

    def search_sales(
        self,
        sale_type: str = "",
        q: str = "",
        name: str = "",
        email: str = "",
        event_id: str = "",
        exact_date: str = "",
        date_from: str = "",
        date_to: str = "",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/sales/search",
            params={
                "type": sale_type,
                "q": q,
                "name": name,
                "email": email,
                "event_id": event_id,
                "date": exact_date,
                "from": date_from,
                "to": date_to,
                "limit": limit,
            },
        )

    def get_sales_latest(
        self,
        event_id: str = "",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/sales/latest",
            params={"event_id": event_id, "limit": limit},
        )

    def get_availability(
        self,
        event_id: str = "",
        q: str = "",
        scope: str = "",
        limit: Optional[int] = None,
    ) -> dict[str, Any]:
        return self._request(
            "GET",
            "/availability",
            params={"event_id": event_id, "q": q, "scope": scope, "limit": limit},
        )

    def sell_cash(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self._request("POST", "/taquilla/sell-cash", json_body=payload)
