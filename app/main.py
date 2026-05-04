"""Application entrypoint."""

from __future__ import annotations

from typing import Optional

import gradio as gr

from app.agent import handle_message
from app.tools.db import (
    spend_by_category_calendar,
    top_vendors_calendar,
    list_pending_receipts,
)


def _format_assistant_response(payload: dict) -> str:
    response = str(payload.get("response", ""))
    citations = payload.get("citations", []) or []
    if citations:
        response += "\n\nCitations: " + ", ".join(str(c) for c in citations)
    return response


def _get_spending_by_category_data(period: str) -> list[list]:
    """Fetch spending by category data and format for dataframe."""
    try:
        rows = spend_by_category_calendar(period=period, n_periods=8)
        if not rows:
            return []
        
        # Format: (period_label, category, total_spend, count)
        formatted = []
        for period_label, category, total_spend, count in rows:
            formatted.append([
                period_label,
                category.capitalize(),
                f"${total_spend:.2f}",
                int(count),
            ])
        return formatted
    except Exception as e:
        return []


def _get_top_vendors_data(period: str) -> list[list]:
    """Fetch top vendors data and format for dataframe."""
    try:
        rows = top_vendors_calendar(period=period, n_periods=8, top_k=5)
        if not rows:
            return []
        
        # Format: (period_label, vendor, total_spend)
        formatted = []
        for period_label, vendor, total_spend in rows:
            formatted.append([
                period_label,
                vendor or "(Unknown)",
                f"${total_spend:.2f}",
            ])
        return formatted
    except Exception as e:
        return []


def _get_pending_receipts_data() -> list[list]:
    """Fetch pending receipts data and format for dataframe."""
    try:
        pending = list_pending_receipts(limit=50)
        if not pending:
            return []
        
        formatted = []
        for item in pending:
            doc_id = item.get("doc_id", "")
            vendor = item.get("vendor") or "(empty)"
            date = item.get("doc_date") or "(empty)"
            total = item.get("total") or "0.00"
            missing = item.get("missing_fields", "none")
            
            # Format total as currency
            if isinstance(total, (int, float)):
                total = f"${total:.2f}"
            
            formatted.append([
                doc_id,
                vendor,
                date,
                total,
                missing,
            ])
        return formatted
    except Exception as e:
        return []


def _refresh_dashboard_data(period: str) -> tuple[list[list], list[list]]:
    """Refresh dashboard data based on selected period."""
    category_data = _get_spending_by_category_data(period)
    vendors_data = _get_top_vendors_data(period)
    return category_data, vendors_data


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
        gr.Markdown("# ReceiptIQ - Expense Management")
        
        with gr.Tabs():
            # ===== Tab 1: Chat =====
            with gr.Tab("Chat"):
                gr.Markdown("Chat with ReceiptIQ assistant or upload a receipt to process.")
                
                chatbot = gr.Chatbot(label="Assistant", height=420)

                with gr.Row():
                    message = gr.Textbox(
                        label="Message",
                        placeholder="Ask about recent receipts, spending, categories, weekly/monthly summaries, pending receipts, or upload a file.",
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

            # ===== Tab 2: Dashboard =====
            with gr.Tab("Dashboard"):
                gr.Markdown("### Spending Analytics")
                
                with gr.Row():
                    period_dropdown = gr.Dropdown(
                        choices=["week", "month"],
                        value="week",
                        label="Period",
                        scale=1,
                    )
                    refresh_btn = gr.Button("🔄 Refresh", variant="primary", scale=1)
                
                gr.Markdown("#### Spending by Category")
                category_table = gr.Dataframe(
                    value=_get_spending_by_category_data("week"),
                    headers=["Period", "Category", "Total", "Count"],
                    interactive=False,
                )
                
                gr.Markdown("#### Top Vendors")
                vendors_table = gr.Dataframe(
                    value=_get_top_vendors_data("week"),
                    headers=["Period", "Vendor", "Total Spend"],
                    interactive=False,
                )
                
                # Refresh button click handler
                def on_dashboard_refresh(period):
                    cat_data, vend_data = _refresh_dashboard_data(period)
                    return cat_data, vend_data
                
                refresh_btn.click(
                    on_dashboard_refresh,
                    [period_dropdown],
                    [category_table, vendors_table],
                )
                
                # Period dropdown change handler
                def on_period_change(period):
                    cat_data, vend_data = _refresh_dashboard_data(period)
                    return cat_data, vend_data
                
                period_dropdown.change(
                    on_period_change,
                    [period_dropdown],
                    [category_table, vendors_table],
                )

            # ===== Tab 3: Pending Receipts =====
            with gr.Tab("Pending Receipts"):
                gr.Markdown("### Pending Receipts")
                gr.Markdown("These receipts are missing critical information (vendor, date, or total). Click the refresh button to update the list.")
                
                with gr.Row():
                    refresh_pending_btn = gr.Button("🔄 Refresh", variant="primary", scale=1)
                
                pending_table = gr.Dataframe(
                    value=_get_pending_receipts_data(),
                    headers=["Doc ID", "Vendor", "Date", "Total", "Missing Fields"],
                    interactive=False,
                )
                
                # Refresh button click handler
                def on_pending_refresh():
                    return _get_pending_receipts_data()
                
                refresh_pending_btn.click(
                    on_pending_refresh,
                    [],
                    [pending_table],
                )

    return demo


def main() -> None:
    app = build_app()
    app.launch(share=True)


if __name__ == "__main__":
    main()

