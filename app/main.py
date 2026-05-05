"""Application entrypoint."""

from __future__ import annotations

from typing import Optional

import gradio as gr

from app.agent import handle_message
from app.tools.db import (
    spend_by_category_calendar,
    top_vendors_calendar,
    list_pending_receipts,
    update_document,
    get_document_by_id,
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
                (category or "uncategorized").capitalize(),
                f"${total_spend:.2f}",
                int(count) if count else 0,
            ])
        return formatted
    except Exception as e:
        print(f"Error in _get_spending_by_category_data: {e}")
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


def _get_pending_receipts_choices() -> list[str]:
    """Get pending receipts as radio button choices."""
    try:
        pending = list_pending_receipts(limit=50)
        if not pending:
            return []
        
        choices = []
        for item in pending:
            doc_id = item.get("doc_id", "")
            vendor = item.get("vendor") or "(empty)"
            total = item.get("total") or "0.00"
            
            # Format as: "Doc #123 - Vendor - $12.34"
            if isinstance(total, (int, float)):
                total_str = f"${total:.2f}"
            else:
                total_str = str(total)
            
            choice_label = f"Doc #{doc_id} - {vendor} - {total_str}"
            choices.append(choice_label)
        
        return choices
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


def _get_pending_receipt_by_index(index: int) -> dict:
    """Get a specific pending receipt by index."""
    try:
        pending = list_pending_receipts(limit=50)
        if index < 0 or index >= len(pending):
            return {}
        return pending[index]
    except Exception as e:
        return {}


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
                
                # Check initial state
                pending_choices = _get_pending_receipts_choices()
                has_pending = len(pending_choices) > 0
                
                # Refresh button (always visible)
                with gr.Row():
                    refresh_pending_btn = gr.Button("🔄 Refresh List", variant="primary", scale=1)
                
                # Message for when no pending receipts
                no_pending_msg = gr.Markdown(
                    "## ✅ No Pending Receipts\nAll receipts have been completed!",
                    visible=not has_pending
                )
                
                # Instructions
                instructions_msg = gr.Markdown(
                    "These receipts are missing critical information. Select one from the list below to edit it.",
                    visible=has_pending
                )
                
                # Radio buttons for selecting which receipt to edit (always created, visibility controlled separately)
                pending_radio = gr.Radio(
                    choices=pending_choices,
                    label="Select a Receipt to Edit",
                    info="Choose from pending receipts",
                    visible=has_pending,
                )
                
                # Edit form (initially hidden, shows only when a receipt is selected)
                with gr.Group(visible=False) as edit_form_group:
                    gr.Markdown("#### Edit Pending Receipt")
                    
                    with gr.Row():
                        selected_doc_id = gr.Number(label="Doc ID", interactive=False, scale=3)
                        close_form_btn = gr.Button("✕ Close", variant="secondary", scale=1)
                    
                    with gr.Row():
                        edit_vendor = gr.Textbox(label="Vendor", scale=2)
                        edit_category = gr.Dropdown(
                            choices=["meals", "travel", "supplies", "other"],
                            label="Category",
                            value="other",
                            scale=1,
                        )
                    
                    with gr.Row():
                        edit_date = gr.Textbox(label="Date (YYYY-MM-DD)", scale=1)
                        edit_total = gr.Number(label="Total Amount", scale=1)
                    
                    with gr.Row():
                        save_btn = gr.Button("💾 Save Changes", variant="primary", scale=2)
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    
                    status_msg = gr.Textbox(label="Status", interactive=False, visible=True)
                
                # Handler: When radio selection changes, show form and load data
                def on_radio_select(selected_label):
                    """Load receipt data when radio button selected and show form."""
                    if not selected_label:
                        return 0, "other", "", "", 0, "No receipt selected", gr.update(visible=False)
                    
                    # Find matching receipt by label
                    try:
                        pending = list_pending_receipts(limit=50)
                        for item in pending:
                            doc_id = item.get("doc_id", "")
                            vendor = item.get("vendor") or "(empty)"
                            total = item.get("total") or "0.00"
                            
                            if isinstance(total, (int, float)):
                                total_str = f"${total:.2f}"
                            else:
                                total_str = str(total)
                            
                            choice_label = f"Doc #{doc_id} - {vendor} - {total_str}"
                            
                            if choice_label == selected_label:
                                # Found matching receipt - load and show form
                                doc_id_int = int(doc_id) if doc_id else 0
                                vendor = item.get("vendor") or ""
                                date = item.get("doc_date") or ""
                                total = float(item.get("total", 0)) if item.get("total") else 0
                                category = item.get("category") or "other"
                                
                                return doc_id_int, category, vendor, date, total, f"✓ Loaded doc #{doc_id_int}", gr.update(visible=True)
                        
                        return 0, "other", "", "", 0, "Receipt not found", gr.update(visible=False)
                    except Exception as e:
                        return 0, "other", "", "", 0, f"Error: {str(e)}", gr.update(visible=False)
                
                # Handler: Save changes (always attach)
                def save_pending_changes(doc_id, vendor, category, date, total):
                    """Save edited pending receipt back to database."""
                    if not doc_id or doc_id == 0:
                        return "Please load a receipt first", gr.update(choices=_get_pending_receipts_choices(), value=None)
                    
                    try:
                        doc_id_int = int(doc_id)
                        updates = {}
                        
                        if vendor and vendor.strip():
                            updates["vendor"] = vendor.strip()
                        if category and category.strip():
                            updates["category"] = category.strip()
                        if date and date.strip():
                            updates["doc_date"] = date.strip()
                        if total and total > 0:
                            updates["total"] = float(total)
                        
                        if not updates:
                            return "No changes to save", gr.update(choices=_get_pending_receipts_choices(), value=None)
                        
                        # If vendor, date, and total are all provided, mark as no longer pending
                        has_vendor = updates.get("vendor") or (vendor and vendor.strip())
                        has_date = updates.get("doc_date") or (date and date.strip())
                        has_total = updates.get("total") or (total and total > 0)
                        
                        if has_vendor and has_date and has_total:
                            updates["is_pending"] = False
                        
                        update_document(doc_id_int, updates)
                        # Return updated choices to refresh the radio button
                        return f"✓ Saved doc #{doc_id_int}", gr.update(choices=_get_pending_receipts_choices(), value=None)
                    except Exception as e:
                        return f"Error saving: {str(e)}", gr.update(choices=_get_pending_receipts_choices(), value=None)
                
                save_btn.click(
                    save_pending_changes,
                    [selected_doc_id, edit_vendor, edit_category, edit_date, edit_total],
                    [status_msg, pending_radio],
                )
                
                # Handler: Clear form
                def clear_form():
                    return 0, "other", "", "", 0, "Cleared"
                
                clear_btn.click(
                    clear_form,
                    [],
                    [selected_doc_id, edit_category, edit_vendor, edit_date, edit_total, status_msg],
                )
                
                # Handler: Close form
                def close_form():
                    return None, gr.update(visible=False)
                
                close_form_btn.click(
                    close_form,
                    [],
                    [pending_radio, edit_form_group],
                )
                
                # Refresh button click handler
                def on_pending_refresh():
                    new_choices = _get_pending_receipts_choices()
                    new_has_pending = len(new_choices) > 0
                    return (
                        gr.update(choices=new_choices, value=None, visible=new_has_pending),
                        gr.update(visible=new_has_pending),
                        gr.update(visible=not new_has_pending),
                        gr.update(visible=False),
                        0, "other", "", "", 0, "✓ Refreshed - select a receipt"
                    )
                
                refresh_pending_btn.click(
                    on_pending_refresh,
                    [],
                    [pending_radio, instructions_msg, no_pending_msg, edit_form_group, selected_doc_id, edit_category, edit_vendor, edit_date, edit_total, status_msg],
                )
                
                # Attach radio change handler always (will work when radio choices are populated)
                pending_radio.change(
                    on_radio_select,
                    [pending_radio],
                    [selected_doc_id, edit_category, edit_vendor, edit_date, edit_total, status_msg, edit_form_group],
                )


    return demo


def main() -> None:
    app = build_app()
    app.launch(share=True)


if __name__ == "__main__":
    main()

