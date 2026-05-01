"""Application entrypoint."""

from __future__ import annotations

from typing import Optional

import gradio as gr

from app.agent import handle_message
from app.storage import get_pending_review_receipts, get_receipt, update_receipt


def _format_assistant_response(payload: dict) -> str:
    response = str(payload.get("response", ""))
    citations = payload.get("citations", []) or []
    if citations:
        response += "\n\nCitations: " + ", ".join(str(c) for c in citations)
    return response


def _load_pending_receipts_table() -> tuple[list, list]:
    """Load pending receipts for table display."""
    pending = get_pending_review_receipts()
    if not pending:
        return [], []
    
    # Create table rows
    headers = ["Image", "Missing Fields", "Confidence", "Vendor", "Date", "Total"]
    rows = []
    for receipt in pending:
        missing = ", ".join(receipt.get("missing_fields", []))
        rows.append([
            receipt["image_name"],
            missing,
            f"{receipt.get('confidence_overall', 0):.0%}",
            receipt.get("vendor", "—"),
            receipt.get("date", "—"),
            f"${receipt.get('total', '—')}" if receipt.get('total') is not None else "—",
        ])
    
    return headers, rows


def _on_receipt_select(selected_row: list) -> tuple[str, str, str, str, str, str, str, str]:
    """Load receipt details when row is selected."""
    if not selected_row or not selected_row[0]:
        return "", "", "", "", "", "", "", ""
    
    image_name = selected_row[0]
    receipt = get_receipt(image_name)
    
    if not receipt:
        return "", "", "", "", "", "", "", ""
    
    return (
        image_name,
        receipt.get("vendor", ""),
        receipt.get("date", ""),
        receipt.get("category", ""),
        str(receipt.get("subtotal", "")),
        str(receipt.get("tax", "")),
        str(receipt.get("total", "")),
        receipt.get("invoice_number", ""),
    )


def _on_receipt_update(
    image_name: str,
    vendor: str,
    date: str,
    category: str,
    subtotal: str,
    tax: str,
    total: str,
    invoice_number: str,
) -> str:
    """Update receipt and return success message."""
    if not image_name:
        return "❌ No receipt selected"
    
    try:
        updates = {
            "vendor": vendor.strip() if vendor else None,
            "date": date.strip() if date else None,
            "category": category.strip() if category else None,
            "subtotal": float(subtotal) if subtotal.strip() else None,
            "tax": float(tax) if tax.strip() else None,
            "total": float(total) if total.strip() else None,
            "invoice_number": invoice_number.strip() if invoice_number else None,
        }
        
        update_receipt(image_name, updates)
        return f"✅ Updated {image_name} successfully!"
    except ValueError as e:
        return f"❌ Error: Invalid number format. {e}"
    except Exception as e:
        return f"❌ Error updating receipt: {e}"


def _refresh_pending_table() -> tuple[list, list]:
    """Refresh the pending receipts table."""
    return _load_pending_receipts_table()


def _on_submit(
    message: str,
    file_path: Optional[str],
    history: list,
) -> tuple[str, None, list]:
    history = history or []
    message = (message or "").strip()

    if not message and not file_path:
        return "", None, history

    try:
        result = handle_message(message, file_path=file_path)
        assistant_text = _format_assistant_response(result)
    except Exception as exc:
        assistant_text = f"Error processing request: {exc}"

    user_text = message if message else "[uploaded file]"
    
    # Append messages in Gradio's expected format
    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": assistant_text})
    
    return "", None, history


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ReceiptIQ") as demo:
        gr.Markdown("# ReceiptIQ")
        
        with gr.Tabs():
            # Tab 1: Chat interface
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(label="Assistant", height=420)

                with gr.Row():
                    message = gr.Textbox(
                        label="Message",
                        placeholder="Ask to list receipts, vendor spend, duplicates, or upload a file.",
                        scale=4,
                    )
                    file_input = gr.File(
                        label="Receipt/Invoice File",
                        file_types=[".png", ".jpg", ".jpeg", ".pdf"],
                        type="filepath",
                        scale=2,
                    )
                    send_button = gr.Button("Send", variant="primary", scale=1)

                message.submit(_on_submit, [message, file_input, chatbot], [message, file_input, chatbot])
                send_button.click(_on_submit, [message, file_input, chatbot], [message, file_input, chatbot])

            # Tab 2: Pending review receipts
            with gr.Tab("Show me pending review receipts"):
                gr.Markdown("### Receipts Pending Manual Review")
                gr.Markdown("Click on a row to edit the receipt fields. Fill in missing information and click 'Update Receipt'.")
                
                with gr.Row():
                    refresh_btn = gr.Button("🔄 Refresh", variant="secondary")
                
                # Table display
                headers, rows = _load_pending_receipts_table()
                with gr.Row():
                    receipt_table = gr.Dataframe(
                        value=rows,
                        headers=headers,
                        interactive=False,
                        row_count=10,
                    )
                
                gr.Markdown("### Edit Selected Receipt")
                
                with gr.Row():
                    image_name_display = gr.Textbox(label="Image Name", interactive=False)
                
                with gr.Row():
                    vendor_field = gr.Textbox(label="Vendor", placeholder="e.g., Starbucks")
                    date_field = gr.Textbox(label="Date", placeholder="e.g., 01/15/2024 or 2024-01-15")
                
                with gr.Row():
                    category_field = gr.Dropdown(
                        choices=["meals", "travel", "supplies", "other"],
                        label="Category",
                        interactive=True,
                    )
                    invoice_field = gr.Textbox(label="Invoice Number", placeholder="Optional")
                
                with gr.Row():
                    subtotal_field = gr.Number(label="Subtotal ($)", precision=2)
                    tax_field = gr.Number(label="Tax ($)", precision=2)
                    total_field = gr.Number(label="Total ($)", precision=2)
                
                with gr.Row():
                    update_btn = gr.Button("✅ Update Receipt", variant="primary")
                    status_msg = gr.Textbox(label="Status", interactive=False)
                
                # Handle row selection
                receipt_table.select(
                    _on_receipt_select,
                    [receipt_table],
                    [
                        image_name_display,
                        vendor_field,
                        date_field,
                        category_field,
                        subtotal_field,
                        tax_field,
                        total_field,
                        invoice_field,
                    ],
                )
                
                # Handle update
                update_btn.click(
                    _on_receipt_update,
                    [
                        image_name_display,
                        vendor_field,
                        date_field,
                        category_field,
                        subtotal_field,
                        tax_field,
                        total_field,
                        invoice_field,
                    ],
                    [status_msg],
                )
                
                # Handle refresh
                refresh_btn.click(
                    _refresh_pending_table,
                    [],
                    [receipt_table],
                )

    return demo


def main() -> None:
    app = build_app()
    app.launch(share=True)


if __name__ == "__main__":
    main()
