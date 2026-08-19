from __future__ import annotations

import base64
import json
import os
import re
import time
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Literal, Optional

from bs4 import BeautifulSoup
from google import genai
from google.auth.transport.requests import Request
from google.genai import types
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
from pydantic import BaseModel, Field, RootModel

BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "token.json"

SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

# Seguridad operativa: deja True hasta revisar varias corridas.
dry_run = True
MAX_EMAILS = 100
BATCH_SIZE = 20
MODEL_NAME = "gemini-flash-latest"
FALLBACK_MODELS = ["gemini-3.5-flash"]
MAX_RETRIES_PER_MODEL = 3

# Politica de borrado.
DELETE_ONLY_IF_CATEGORY = "Promociones Irrelevantes"
DELETE_MIN_CONFIDENCE = 88
DELETE_OLD_EMAILS_ENABLED = False
DELETE_IF_YEAR_LESS_OR_EQUAL = 2023

PROMO_KEYWORDS = {
    "oferta",
    "ofertas",
    "descuento",
    "descuentos",
    "promo",
    "promocion",
    "promociones",
    "cupon",
    "cupón",
    "off",
    "sale",
    "newsletter",
    "compra ahora",
    "comprá ahora",
    "ultimas horas",
    "últimas horas",
    "aprovecha",
    "aprovechá",
    "rebaja",
    "liquidacion",
    "liquidación",
    "hot sale",
    "cyber",
    "envio gratis",
    "envío gratis",
    "2x1",
    "3x2",
}

PROTECTED_KEYWORDS = {
    "factura",
    "recibo",
    "comprobante",
    "pago",
    "impuesto",
    "invoice",
    "receipt",
    "shared",
    "compartió",
    "compartio",
    "documento",
    "google docs",
    "drive",
    "condiciones del servicio",
    "service terms",
    "password",
    "contraseña",
    "codigo de verificacion",
    "código de verificación",
    "login",
    "security",
    "seguridad",
    "alerta",
    "alert",
    "trabajo",
    "job",
    "interview",
    "entrevista",
    "application",
    "aplicacion",
    "aplicación",
    "reservation",
    "reserva",
    "order",
    "pedido",
    "shipment",
    "envio",
    "envío",
}


class EmailClassification(BaseModel):
    id: str
    categoria: Literal[
        "Importante",
        "Facturas y Comprobantes",
        "Promociones Irrelevantes",
        "Notificaciones de Sistemas",
    ]
    confianza_porcentaje: int = Field(ge=0, le=100)
    eliminar: bool


class EmailClassificationBatch(RootModel[list[EmailClassification]]):
    pass


CLASSIFICATION_PROMPT = """
Sos un clasificador estricto de correos de Gmail que trabaja por lotes.

Debes devolver EXCLUSIVAMENTE un JSON valido.
La salida debe ser un array JSON, un objeto por cada correo recibido, manteniendo el campo `id` de entrada.

Schema de cada objeto:
[
  {
    "id": "string",
    "categoria": "Importante" | "Facturas y Comprobantes" | "Promociones Irrelevantes" | "Notificaciones de Sistemas",
    "confianza_porcentaje": numero entero entre 0 y 100,
    "eliminar": true | false
  }
]

Reglas:
- Importante: correos personales, laborales o utiles que requieren seguimiento.
- Facturas y Comprobantes: facturas, tickets, recibos, pagos y comprobantes.
- Promociones Irrelevantes: publicidad agresiva, ofertas poco utiles, newsletters comerciales descartables.
- Notificaciones de Sistemas: accesos, resets, alertas automaticas, monitoreo y mensajes de plataformas.

Casos que NO deben ir a `Promociones Irrelevantes` salvo evidencia comercial muy clara:
- documentos compartidos, archivos de Drive, mensajes de estudio o trabajo
- cambios de terminos, condiciones o politicas de servicios que ya usas
- seguridad, accesos, codigos, reseteo de password
- comprobantes, pedidos, reservas, entrevistas, postulaciones y envios

Regla estricta de eliminacion:
- `eliminar` debe ser true SOLO si la categoria es `Promociones Irrelevantes` y la confianza_porcentaje es mayor a 88.
- En cualquier otro caso, `eliminar` debe ser false.

No agregues texto adicional.
""".strip()


def get_env_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Falta GEMINI_API_KEY. En PowerShell define primero: "
            "$env:GEMINI_API_KEY='tu_api_key'"
        )
    return api_key


def get_gmail_credentials() -> Credentials:
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"No existe {TOKEN_FILE}")

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    if not creds.valid:
        raise RuntimeError(
            "token.json no es valido para gmail.modify. Regeneralo con generate_gmail_token.py."
        )

    return creds


def build_gmail_service():
    creds = get_gmail_credentials()
    service = build("gmail", "v1", credentials=creds, cache_discovery=False)
    service.users().getProfile(userId="me").execute()
    return service


def get_header_value(headers: list[dict], name: str, default: str = "") -> str:
    for header in headers or []:
        if header.get("name", "").lower() == name.lower():
            return header.get("value", default)
    return default


def decode_base64url(data: str) -> str:
    if not data:
        return ""
    padding = "=" * (-len(data) % 4)
    raw = base64.urlsafe_b64decode(data + padding)
    return raw.decode("utf-8", errors="replace")


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", text or "").strip()


def html_to_text(html: str) -> str:
    if not html:
        return ""
    return clean_text(BeautifulSoup(html, "html.parser").get_text(" "))


def extract_message_text(payload: Optional[dict]) -> str:
    if not payload:
        return ""

    mime_type = payload.get("mimeType", "")
    body_data = payload.get("body", {}).get("data")

    if body_data:
        decoded = decode_base64url(body_data)
        if mime_type == "text/html":
            return html_to_text(decoded)
        return clean_text(decoded)

    for part in payload.get("parts", []) or []:
        text = extract_message_text(part)
        if text:
            return text

    return ""


def normalize_date(date_value: str) -> str:
    if not date_value:
        return ""
    try:
        return parsedate_to_datetime(date_value).isoformat()
    except Exception:
        return date_value


def fetch_recent_inbox_messages(service, max_results: int = 30) -> list[dict]:
    response = service.users().messages().list(
        userId="me",
        labelIds=["INBOX"],
        maxResults=max_results,
    ).execute()

    detailed_messages = []
    for item in response.get("messages", []):
        message_id = item.get("id")
        if not message_id:
            continue

        try:
            msg = service.users().messages().get(
                userId="me",
                id=message_id,
                format="full",
            ).execute()

            payload = msg.get("payload", {})
            headers = payload.get("headers", [])
            body_text = extract_message_text(payload)
            snippet = clean_text(msg.get("snippet", ""))
            preview = (body_text or snippet)[:500]

            detailed_messages.append(
                {
                    "id": msg.get("id", ""),
                    "date": normalize_date(get_header_value(headers, "Date")),
                    "from": get_header_value(headers, "From", "(Sin remitente)"),
                    "subject": get_header_value(headers, "Subject", "(Sin asunto)"),
                    "snippet": preview,
                }
            )
        except Exception as exc:
            print(f"[WARN] No se pudo leer el mail {message_id}: {exc}")

    return detailed_messages


def chunk_list(items: list[dict], chunk_size: int) -> list[list[dict]]:
    return [items[index : index + chunk_size] for index in range(0, len(items), chunk_size)]


def normalize_for_match(*parts: str) -> str:
    return clean_text(" ".join(part for part in parts if part).casefold())


def keyword_hits(text: str, keywords: set[str]) -> set[str]:
    return {keyword for keyword in keywords if keyword in text}


def extract_email_year(date_value: str) -> Optional[int]:
    if not date_value:
        return None
    match = re.match(r"^(\d{4})-", date_value)
    if match:
        return int(match.group(1))
    fallback = re.search(r"\b(20\d{2})\b", date_value)
    if fallback:
        return int(fallback.group(1))
    return None


def decide_trash_action(email_data: dict, decision: EmailClassification) -> tuple[bool, str]:
    email_year = extract_email_year(email_data.get("date", ""))
    if (
        DELETE_OLD_EMAILS_ENABLED
        and email_year is not None
        and email_year <= DELETE_IF_YEAR_LESS_OR_EQUAL
    ):
        return True, f"antiguedad: año {email_year} <= {DELETE_IF_YEAR_LESS_OR_EQUAL}"

    normalized_text = normalize_for_match(
        email_data.get("subject", ""),
        email_data.get("from", ""),
        email_data.get("snippet", ""),
    )
    promo_hits = keyword_hits(normalized_text, PROMO_KEYWORDS)
    protected_hits = keyword_hits(normalized_text, PROTECTED_KEYWORDS)

    if protected_hits:
        return False, f"protegido por senales utiles: {', '.join(sorted(protected_hits)[:3])}"

    if decision.categoria != DELETE_ONLY_IF_CATEGORY:
        return False, "categoria no eliminable"

    if decision.confianza_porcentaje < DELETE_MIN_CONFIDENCE:
        return False, f"confianza insuficiente ({decision.confianza_porcentaje}%)"

    if len(promo_hits) >= 2:
        return True, f"promo clara por keywords: {', '.join(sorted(promo_hits)[:3])}"

    if decision.confianza_porcentaje >= 96 and len(promo_hits) >= 1:
        return True, "promo con confianza muy alta"

    return False, "sin suficientes senales comerciales para borrar"


def classify_email_batch_with_gemini(
    client: genai.Client, email_batch: list[dict]
) -> list[EmailClassification]:
    payload = [
        {
            "id": email_data["id"],
            "date": email_data["date"],
            "from": email_data["from"],
            "subject": email_data["subject"],
            "snippet": email_data["snippet"],
        }
        for email_data in email_batch
    ]
    models_to_try = [MODEL_NAME, *FALLBACK_MODELS]
    last_error: Exception | None = None
    batch_label = f"lote de {len(email_batch)} correos"

    for model_name in models_to_try:
        for attempt in range(1, MAX_RETRIES_PER_MODEL + 1):
            try:
                response = client.models.generate_content(
                    model=model_name,
                    contents=[
                        {"role": "user", "parts": [{"text": CLASSIFICATION_PROMPT}]},
                        {
                            "role": "user",
                            "parts": [
                                {
                                    "text": "Clasifica este lote de correos y aplica la regla estricta de eliminacion:\n"
                                    + json.dumps(payload, ensure_ascii=False)
                                }
                            ],
                        },
                    ],
                    config=types.GenerateContentConfig(
                        temperature=0,
                        response_mime_type="application/json",
                        response_schema=EmailClassificationBatch,
                    ),
                )

                if response.parsed:
                    return response.parsed.root

                raw_text = getattr(response, "text", "") or ""
                return EmailClassificationBatch.model_validate_json(raw_text).root
            except Exception as exc:
                last_error = exc
                error_text = str(exc)
                is_retryable = (
                    "503" in error_text
                    or "UNAVAILABLE" in error_text
                    or "429" in error_text
                    or "RESOURCE_EXHAUSTED" in error_text
                )
                if not is_retryable:
                    raise

                if attempt < MAX_RETRIES_PER_MODEL:
                    wait_seconds = attempt * 2
                    retry_match = re.search(r"retry in ([0-9]+(?:\.[0-9]+)?)s", error_text)
                    if retry_match:
                        wait_seconds = max(wait_seconds, int(float(retry_match.group(1))) + 1)
                    print(
                        f"[WARN][Gemini] {batch_label}: "
                        f"{model_name} saturado. Reintentando en {wait_seconds}s..."
                    )
                    time.sleep(wait_seconds)

        print(f"[WARN][Gemini] Fallback al siguiente modelo despues de agotar {model_name}.")

    assert last_error is not None
    raise last_error


def main() -> None:
    gmail_service = build_gmail_service()
    gemini_client = genai.Client(api_key=get_env_api_key())

    print(f"dry_run = {dry_run}")
    print(f"Modelo Gemini = {MODEL_NAME}")
    print(f"MAX_EMAILS = {MAX_EMAILS}")
    print(f"BATCH_SIZE = {BATCH_SIZE}")

    emails = fetch_recent_inbox_messages(gmail_service, max_results=MAX_EMAILS)
    print(f"Correos listos para clasificar: {len(emails)}")

    processed = 0
    simulated_trash = 0
    real_trash = 0
    errors = 0
    quota_exhausted = False

    for email_batch in chunk_list(emails, BATCH_SIZE):
        if quota_exhausted:
            print("[WARN][Gemini] Se detiene el procesamiento porque se agoto la cuota del modelo.")
            break

        try:
            decisions = classify_email_batch_with_gemini(gemini_client, email_batch)
            decisions_by_id = {decision.id: decision for decision in decisions}

            for email_data in email_batch:
                processed += 1
                subject = email_data.get("subject", "(Sin asunto)")
                decision = decisions_by_id.get(email_data["id"])
                if not decision:
                    errors += 1
                    print(f"[ERROR][Gemini/Parseo] {subject}: no volvio clasificacion para este id.")
                    continue

                should_delete, action_reason = decide_trash_action(email_data, decision)

                action = "MANTENER"
                if should_delete and dry_run:
                    action = "MOVER A PAPELERA (Simulado)"
                    simulated_trash += 1
                elif should_delete and not dry_run:
                    gmail_service.users().messages().trash(
                        userId="me",
                        id=email_data["id"],
                    ).execute()
                    action = "MOVER A PAPELERA"
                    real_trash += 1

                print(
                    f"[{subject}] -> {decision.categoria} "
                    f"({decision.confianza_porcentaje}%) -> Accion: {action} | Motivo: {action_reason}"
                )
        except HttpError as exc:
            errors += len(email_batch)
            print(f"[ERROR][Gmail] lote de {len(email_batch)} correos: {exc}")
        except Exception as exc:
            errors += len(email_batch)
            print(f"[ERROR][Gemini/Parseo] lote de {len(email_batch)} correos: {exc}")
            error_text = str(exc)
            if "429" in error_text or "RESOURCE_EXHAUSTED" in error_text:
                quota_exhausted = True

    print("\nResumen")
    print(f"- Procesados: {processed}")
    print(f"- Simulados a papelera: {simulated_trash}")
    print(f"- Enviados realmente a papelera: {real_trash}")
    print(f"- Errores: {errors}")


if __name__ == "__main__":
    main()
