from __future__ import annotations

import argparse
import base64
import csv
import re
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Optional

from bs4 import BeautifulSoup
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from joblib import dump, load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent
TOKEN_FILE = BASE_DIR / "token.json"
SCOPES = ["https://www.googleapis.com/auth/gmail.modify"]

DATASET_FILE = BASE_DIR / "gmail_training_data.csv"
MODEL_FILE = BASE_DIR / "gmail_classifier.joblib"

LABELS = [
    "Importante",
    "Facturas y Comprobantes",
    "Promociones Irrelevantes",
    "Notificaciones de Sistemas",
]

LABEL_SHORTCUTS = {
    "1": "Importante",
    "2": "Facturas y Comprobantes",
    "3": "Promociones Irrelevantes",
    "4": "Notificaciones de Sistemas",
}

DELETE_MIN_CONFIDENCE = 0.70
dry_run = True

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
    "sale",
    "% off",
    "dto",
    "newsletter",
    "desuscribirse",
    "unsubscribe",
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
    "order",
    "pedido",
    "shipment",
}

ALWAYS_KEEP_SENDERS = {
    "tryhackme.com",
    "docs.google.com",
    "google.com",
    "mercadopago.com",
}


def get_gmail_credentials() -> Credentials:
    if not TOKEN_FILE.exists():
        raise FileNotFoundError(f"No existe {TOKEN_FILE}")

    creds = Credentials.from_authorized_user_file(str(TOKEN_FILE), SCOPES)
    if creds.expired and creds.refresh_token:
        creds.refresh(Request())
        TOKEN_FILE.write_text(creds.to_json(), encoding="utf-8")

    if not creds.valid:
        raise RuntimeError("token.json no es valido para gmail.modify.")

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


def fetch_recent_inbox_messages(service, max_results: int = 50) -> list[dict]:
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


def text_features(email_data: dict) -> str:
    return " \n ".join(
        [
            email_data.get("from", ""),
            email_data.get("subject", ""),
            email_data.get("snippet", ""),
        ]
    )


def normalize_for_match(*parts: str) -> str:
    return clean_text(" ".join(part for part in parts if part).casefold())


def keyword_hits(text: str, keywords: set[str]) -> set[str]:
    return {keyword for keyword in keywords if keyword in text}


def heuristic_label(email_data: dict) -> tuple[str, str]:
    normalized_text = normalize_for_match(
        email_data.get("subject", ""),
        email_data.get("from", ""),
        email_data.get("snippet", ""),
    )
    promo_hits = keyword_hits(normalized_text, PROMO_KEYWORDS)
    protected_hits = keyword_hits(normalized_text, PROTECTED_KEYWORDS)

    system_hits = keyword_hits(
        normalized_text,
        {
            "security",
            "seguridad",
            "password",
            "contraseña",
            "codigo de verificacion",
            "código de verificación",
            "alerta",
            "alert",
        },
    )
    invoice_hits = keyword_hits(
        normalized_text,
        {
            "factura",
            "recibo",
            "comprobante",
            "invoice",
            "receipt",
            "pedido",
            "order",
        },
    )

    if invoice_hits:
        return "Facturas y Comprobantes", f"keywords de comprobante: {', '.join(sorted(invoice_hits)[:3])}"

    if system_hits:
        return "Notificaciones de Sistemas", f"keywords de sistema: {', '.join(sorted(system_hits)[:3])}"

    if len(promo_hits) >= 2:
        return "Promociones Irrelevantes", f"keywords promocionales: {', '.join(sorted(promo_hits)[:3])}"

    if protected_hits:
        return "Importante", f"keywords protegidas: {', '.join(sorted(protected_hits)[:3])}"

    sender = normalized_text
    if "noreply" in sender or "no-reply" in sender:
        return "Notificaciones de Sistemas", "remitente noreply/no-reply"

    return "Importante", "fallback conservador"


def heuristic_delete_guard(email_data: dict, predicted_label: str, confidence: float) -> tuple[bool, str]:
    sender = (email_data.get("from", "") or "").casefold()
    keep_sender_hits = [domain for domain in ALWAYS_KEEP_SENDERS if domain in sender]
    if keep_sender_hits:
        return False, f"remitente protegido: {keep_sender_hits[0]}"

    normalized_text = normalize_for_match(
        email_data.get("subject", ""),
        email_data.get("from", ""),
        email_data.get("snippet", ""),
    )
    promo_hits = keyword_hits(normalized_text, PROMO_KEYWORDS)
    protected_hits = keyword_hits(normalized_text, PROTECTED_KEYWORDS)

    if protected_hits:
        return False, f"protegido por senales utiles: {', '.join(sorted(protected_hits)[:3])}"

    # Si las señales comerciales son claras, dejamos que la heurística mande
    # aunque el clasificador todavía no tenga confianza alta.
    if len(promo_hits) >= 2:
        return True, f"promo obvia por keywords: {', '.join(sorted(promo_hits)[:3])}"

    if predicted_label != "Promociones Irrelevantes":
        return False, "categoria no eliminable"

    if confidence < DELETE_MIN_CONFIDENCE:
        return False, f"confianza insuficiente ({confidence:.0%})"

    if len(promo_hits) >= 2:
        return True, f"promo clara por keywords: {', '.join(sorted(promo_hits)[:3])}"

    if confidence >= 0.85 and len(promo_hits) >= 1:
        return True, "promo con confianza suficiente"

    return False, "sin suficientes senales comerciales para borrar"


def ensure_dataset_exists() -> None:
    if DATASET_FILE.exists():
        return

    with DATASET_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "date", "from", "subject", "snippet", "label"],
        )
        writer.writeheader()


def export_for_labeling(limit: int) -> None:
    ensure_dataset_exists()
    service = build_gmail_service()
    emails = fetch_recent_inbox_messages(service, max_results=limit)

    existing_ids = set()
    with DATASET_FILE.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            existing_ids.add(row["id"])

    new_rows = [email for email in emails if email["id"] not in existing_ids]

    if not new_rows:
        print("No hay correos nuevos para exportar.")
        return

    with DATASET_FILE.open("a", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "date", "from", "subject", "snippet", "label"],
        )
        for row in new_rows:
            writer.writerow({**row, "label": ""})

    print(f"Exportados {len(new_rows)} correos a {DATASET_FILE.name}")
    print("Completa la columna 'label' con una de estas categorias:")
    print(", ".join(LABELS))


def load_labeled_rows() -> list[dict]:
    ensure_dataset_exists()
    rows = []
    with DATASET_FILE.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            label = row.get("label", "").strip()
            if label in LABELS:
                rows.append(row)
    return rows


def load_all_rows() -> list[dict]:
    ensure_dataset_exists()
    with DATASET_FILE.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def save_all_rows(rows: list[dict]) -> None:
    with DATASET_FILE.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["id", "date", "from", "subject", "snippet", "label"],
        )
        writer.writeheader()
        writer.writerows(rows)


def prelabel_dataset(limit: int, overwrite: bool) -> None:
    export_for_labeling(limit=limit)
    rows = load_all_rows()
    updated = 0

    for row in rows:
        current_label = row.get("label", "").strip()
        if current_label in LABELS and not overwrite:
            continue

        suggested_label, reason = heuristic_label(row)
        row["label"] = suggested_label
        updated += 1
        print(f"[{row.get('subject', '')}] -> {suggested_label} | Motivo: {reason}")

    save_all_rows(rows)
    print(f"Pre-etiquetados {updated} correos en {DATASET_FILE.name}")


def interactive_label(limit: int) -> None:
    export_for_labeling(limit=limit)
    rows = load_all_rows()

    unlabeled_rows = [row for row in rows if row.get("label", "").strip() not in LABELS]
    if not unlabeled_rows:
        print("No hay correos pendientes de etiquetar.")
        return

    print("Etiquetado interactivo")
    print("1 = Importante")
    print("2 = Facturas y Comprobantes")
    print("3 = Promociones Irrelevantes")
    print("4 = Notificaciones de Sistemas")
    print("s = saltar")
    print("q = salir\n")

    updated = False

    for row in rows:
        if row.get("label", "").strip() in LABELS:
            continue

        print("=" * 80)
        print(f"Fecha: {row.get('date', '')}")
        print(f"From: {row.get('from', '')}")
        print(f"Subject: {row.get('subject', '')}")
        print(f"Snippet: {row.get('snippet', '')}")
        print("-" * 80)

        while True:
            choice = input("Etiqueta [1/2/3/4], s=saltar, q=salir: ").strip().lower()
            if choice in LABEL_SHORTCUTS:
                row["label"] = LABEL_SHORTCUTS[choice]
                updated = True
                print(f"Guardado: {row['label']}\n")
                break
            if choice == "s":
                print("Saltado\n")
                break
            if choice == "q":
                save_all_rows(rows)
                print("Etiquetado interrumpido. Cambios guardados.")
                return
            print("Opcion invalida.")

    if updated:
        save_all_rows(rows)
        print("Etiquetado completado. Cambios guardados.")
    else:
        print("No hubo cambios.")


def train_model(test_size: float) -> None:
    rows = load_labeled_rows()
    if len(rows) < 20:
        raise RuntimeError("Necesitas al menos 20 correos etiquetados para entrenar algo util.")

    texts = [text_features(row) for row in rows]
    labels = [row["label"] for row in rows]

    X_train, X_test, y_train, y_test = train_test_split(
        texts,
        labels,
        test_size=test_size,
        random_state=42,
        stratify=labels,
    )

    pipeline = Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    lowercase=True,
                    strip_accents="unicode",
                    ngram_range=(1, 2),
                    min_df=1,
                    max_features=20000,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    print("Reporte de validacion")
    print(classification_report(y_test, predictions, zero_division=0))

    dump(pipeline, MODEL_FILE)
    print(f"Modelo guardado en {MODEL_FILE.name}")


def predict_inbox(limit: int) -> None:
    if not MODEL_FILE.exists():
        raise FileNotFoundError("No existe gmail_classifier.joblib. Entrena primero el modelo.")

    service = build_gmail_service()
    model = load(MODEL_FILE)
    emails = fetch_recent_inbox_messages(service, max_results=limit)

    for email_data in emails:
        text = text_features(email_data)
        predicted_label = model.predict([text])[0]
        probabilities = model.predict_proba([text])[0]
        classes = list(model.classes_)
        confidence = float(probabilities[classes.index(predicted_label)])

        should_delete, reason = heuristic_delete_guard(email_data, predicted_label, confidence)

        action = "MANTENER"
        if should_delete and dry_run:
            action = "MOVER A PAPELERA (Simulado)"
        elif should_delete and not dry_run:
            service.users().messages().trash(userId="me", id=email_data["id"]).execute()
            action = "MOVER A PAPELERA"

        print(
            f"[{email_data['subject']}] -> {predicted_label} ({confidence:.0%}) -> "
            f"Accion: {action} | Motivo: {reason}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline local de clasificacion de Gmail")
    subparsers = parser.add_subparsers(dest="command", required=True)

    export_parser = subparsers.add_parser("export", help="Exporta correos a CSV para etiquetar")
    export_parser.add_argument("--limit", type=int, default=100)

    prelabel_parser = subparsers.add_parser("prelabel", help="Pre-etiqueta el CSV con reglas")
    prelabel_parser.add_argument("--limit", type=int, default=100)
    prelabel_parser.add_argument("--overwrite", action="store_true")

    train_parser = subparsers.add_parser("train", help="Entrena el clasificador local")
    train_parser.add_argument("--test-size", type=float, default=0.25)

    label_parser = subparsers.add_parser("label", help="Etiquetado interactivo en terminal")
    label_parser.add_argument("--limit", type=int, default=100)

    predict_parser = subparsers.add_parser("predict", help="Clasifica el inbox con el modelo local")
    predict_parser.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()

    if args.command == "export":
        export_for_labeling(limit=args.limit)
    elif args.command == "prelabel":
        prelabel_dataset(limit=args.limit, overwrite=args.overwrite)
    elif args.command == "label":
        interactive_label(limit=args.limit)
    elif args.command == "train":
        train_model(test_size=args.test_size)
    elif args.command == "predict":
        predict_inbox(limit=args.limit)


if __name__ == "__main__":
    main()
