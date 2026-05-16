"""Application entrypoint."""

from __future__ import annotations

import logging
import re
import traceback
import os
import io
import base64
from typing import Optional

import gradio as gr

from PIL import Image

from app.agent import handle_message
from app.tools.db import (
    spend_by_category_calendar,
    top_vendors_calendar,
    list_pending_receipts,
    update_document,
    get_document_by_id,
    get_edit_history,
    get_doc_candidates,
    integrity_check
)


# Configure logger
logger = logging.getLogger(__name__)

# Global debug flag for receipt processing logs
# Set to True to enable detailed logging, False to disable
DEBUG_LOGS = True

def log_receipt(message: str, doc_id: int = None):
    """Log receipt-related messages when DEBUG_LOGS is enabled."""
    if DEBUG_LOGS:
        prefix = f"[Receipt #{doc_id}]" if doc_id else "[Receipt]"
        print(f"📋 {prefix} {message}")
        logger.info(f"{prefix} {message}")


def _render_preview(file_path: Optional[str]):
    """Return a PIL image preview for images or first page of PDFs."""
    if not file_path:
        return None

    ext = os.path.splitext(file_path)[1].lower()

    if ext in [".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"]:
        try:
            return Image.open(file_path).convert("RGB")
        except Exception:
            return None

    if ext == ".pdf":
        try:
            import pypdfium2 as pdfium  # type: ignore
            pdf = pdfium.PdfDocument(file_path)
            page = pdf[0]
            pil_image = page.render(scale=2).to_pil()
            return pil_image.convert("RGB")
        except Exception as e:
            logger.warning(f"PDF preview error: {e}")
            return None

    return None


def _pil_to_markdown_clickable(img: Image.Image, thumb_width: int = 320) -> str:
    """Clickable thumbnail: shows small image in chat; click opens full size."""
    # Full-size data URI
    buf_full = io.BytesIO()
    img.save(buf_full, format="PNG")
    full_b64 = base64.b64encode(buf_full.getvalue()).decode("utf-8")
    full_uri = f"data:image/png;base64,{full_b64}"

    # Thumbnail (keep aspect ratio)
    w, h = img.size
    if w > thumb_width:
        new_h = int(h * (thumb_width / float(w)))
        thumb = img.resize((thumb_width, new_h))
    else:
        thumb = img

    buf_thumb = io.BytesIO()
    thumb.save(buf_thumb, format="PNG")
    thumb_b64 = base64.b64encode(buf_thumb.getvalue()).decode("utf-8")
    thumb_uri = f"data:image/png;base64,{thumb_b64}"

    # Markdown clickable thumbnail
    return f"[![]({thumb_uri})]({full_uri})"




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
):
    """Handle chat submit. history is a list of dict messages: {'role','content'}."""
    history = history or []
    message = (message or "").strip()

    # If history accidentally contains tuples from old runs, reset to avoid mixed formats
    if history and isinstance(history[0], tuple):
        history = []

    if not message and not file_path:
        return "", None, history

    # Build user bubble (text + optional inline image/PDF preview)
    user_lines = []
    if message:
        user_lines.append(message)

    if file_path:
        user_lines.append(f"**Uploaded:** {os.path.basename(file_path)}")
        preview_img = _render_preview(file_path)
        if preview_img is not None:
            user_lines.append(_pil_to_markdown_clickable(preview_img, thumb_width=320))

    user_bubble = "\n\n".join(user_lines) if user_lines else "[uploaded file]"

    try:
        result = handle_message(message, file_path=file_path)
        assistant_text = _format_assistant_response(result)
    except Exception as exc:
        assistant_text = f"Error processing request: {exc}"

    # ✅ Messages format (what Gradio expects)
    history.append({"role": "user", "content": user_bubble})
    history.append({"role": "assistant", "content": assistant_text})

    return "", None, history
 
def render_edit_history(doc_id: str) -> str:
    """Return markdown table of edit history for a given doc_id."""
    try:
        if not doc_id or not str(doc_id).strip().isdigit():
            return "Enter a valid doc_id to view edit history."

        from app.tools.db import get_edit_history  # adjust import if your db API differs

        rows = get_edit_history(int(doc_id))
        if not rows:
            return f"No edit history found for doc_id={doc_id}."

        md = "### Edit History\n\n"
        md += "| edited_at | field | old | new |\n"
        md += "|---|---|---|---|\n"
        for r in rows:
            md += f"| {r.get('edited_at','')} | {r.get('field','')} | {r.get('old','')} | {r.get('new','')} |\n"
        return md
    except Exception as e:
        return f"Error loading history: {e}"

def build_app() -> gr.Blocks:
    """Build and return the Gradio application."""
    with gr.Blocks(title="ReceiptIQ") as demo:
        gr.Markdown("# ReceiptIQ - Expense Management")
        
        with gr.Tabs():
            # ===== Tab 1: Chat =====
            with gr.Tab("Chat"):
                gr.Markdown("Chat with ReceiptIQ assistant or upload a receipt to process.")
                
                chatbot = gr.Chatbot(label="Assistant", height=650)

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
                gr.Markdown("*Debug logging enabled - see terminal for detailed processing info*")
                
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
                        edit_vendor = gr.Dropdown(label="Vendor (choose from candidates)", choices=[], value=None, scale=2)
                        edit_category = gr.Dropdown(
                            choices=["meals", "travel", "supplies", "other"],
                            label="Category",
                            value="other",
                            scale=1,
                        )
                    
                    with gr.Row():
                        edit_date = gr.Dropdown(label="Date (choose from candidates)", choices=[], value=None, scale=1)
                        edit_total = gr.Dropdown(label="Total (choose from candidates)", choices=[], value=None, scale=1)
                    
                    with gr.Row():
                        save_btn = gr.Button("💾 Save Changes", variant="primary", scale=2)
                        view_history_btn = gr.Button("View Edit History")
                        history_md = gr.Markdown()
                        clear_btn = gr.Button("🗑️ Clear", variant="secondary", scale=1)
                    
                    status_msg = gr.Textbox(label="Status", interactive=False, visible=True)
                
                # State to track if confirmation is needed for high-risk edit
                confirm_state = gr.State({"needs_confirm": False, "pending_updates": None})
                
                # Handler: When radio selection changes, show form and load data
                def on_radio_select(selected_label):
                    """Load receipt data when radio button selected and show form."""
                    if not selected_label:
                        log_receipt("No receipt selected")
                        return 0, "other", gr.update(choices=["(keep pending / none of these)"], value=None), gr.update(choices=["(keep pending / none of these)"], value=None), gr.update(choices=["(keep pending / none of these)"], value=None), "No receipt selected", gr.update(visible=False)
                    
                    log_receipt(f"LOAD OPERATION STARTED - User selected: {selected_label}")
                    
                    # Find matching receipt by label
                    try:
                        pending = list_pending_receipts(limit=50)
                        log_receipt(f"Fetched {len(pending)} pending receipts")
                        
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
                                
                                log_receipt(f"RECEIPT LOADED - ID: {doc_id_int}", doc_id_int)
                                log_receipt(f"Vendor: '{vendor}'", doc_id_int)
                                log_receipt(f"Date: '{date}'", doc_id_int)
                                log_receipt(f"Total: ${total:.2f}", doc_id_int)
                                log_receipt(f"Category: '{category}'", doc_id_int)
                                log_receipt(f"All fields: {item}", doc_id_int)
                                log_receipt(f"LOAD OPERATION COMPLETED - Form ready for editing\n{'='*60}\n", doc_id_int)
                                # Load candidates for dropdown-only correction UI
                                cands = get_doc_candidates(doc_id_int) or {}
                                
                                def fmt_choices(field: str):
                                    opts = ["(keep pending / none of these)"]
                                    for item in (cands.get(field) or [])[:8]:
                                        if isinstance(item, dict):
                                            val = str(item.get("value") or "").strip()
                                            ev = str(item.get("evidence") or "").strip()
                                            if val:
                                                label = f"{val} — {ev}" if ev and ev != val else val
                                                if label not in opts:
                                                    opts.append(label)
                                    return opts
                                
                                vendor_choices = fmt_choices("vendor")
                                date_choices = fmt_choices("date")
                                
                                total_choices = ["(keep pending / none of these)"]
                                for item in (cands.get("total") or [])[:8]:
                                    if isinstance(item, dict) and item.get("value") is not None:
                                        lab = str(item.get("label") or "amount")
                                        val = item.get("value")
                                        ev = str(item.get("evidence") or "").strip()
                                        base = f"{lab}: {val}"
                                        label = f"{base} — {ev}" if ev and ev != base else base
                                        if label not in total_choices:
                                            total_choices.append(label)
                                
                                vendor_val = vendor if vendor else None
                                date_val = date if date else None
                                total_val = None
                                if isinstance(total, (int, float)) and total > 0:
                                    total_val = f"total: {float(total)}"
                                
                                                                
                                return (
                                    doc_id_int,
                                    category,
                                    gr.update(choices=vendor_choices, value=vendor_val),
                                    gr.update(choices=date_choices, value=date_val),
                                    gr.update(choices=total_choices, value=total_val),
                                    f"✓ Loaded doc #{doc_id_int} (choose candidates to fill missing fields)",
                                    gr.update(visible=True),
                                )
                        
                        log_receipt(f"❌ Receipt not found for label: {selected_label}")
                        return 0, "other", "", "", 0, "Receipt not found", gr.update(visible=False)
                    except Exception as e:
                        log_receipt(f"❌ ERROR loading receipt: {str(e)}")
                        return 0, "other", "", "", 0, f"Error: {str(e)}", gr.update(visible=False)
                
                # Handler: Save changes with high-risk edit warning
                def save_pending_changes(doc_id, vendor, category, date, total, confirm_data):
                    """Save edited pending receipt with high-risk edit warning.
                    
                    Checks if total changed by >30% and requires confirmation before saving.
                    After save, shows integrity check status.
                    """
                    if not doc_id or doc_id == 0:
                        log_receipt("No document loaded")
                        return (
                            "Please load a receipt first",
                            gr.update(choices=_get_pending_receipts_choices(), value=None),
                            {"needs_confirm": False, "pending_updates": None}
                        )
                    
                    try:
                        doc_id_int = int(doc_id)
                        log_receipt(f"SAVE OPERATION STARTED", doc_id_int)
                        log_receipt(f"Input - Vendor: '{vendor}', Category: '{category}', Date: '{date}', Total: {total}", doc_id_int)
                        
                        # Build updates dict
                        updates = {}
                        if vendor and isinstance(vendor, str) and not vendor.startswith("(keep pending"):
                            updates["vendor"] = vendor.split(" — ", 1)[0].strip()
                        if category and category.strip():
                            updates["category"] = category.strip()
                        if date and isinstance(date, str) and not date.startswith("(keep pending"):
                            updates["doc_date"] = date.split(" — ", 1)[0].strip()
                        if total and isinstance(total, str) and not total.startswith("(keep pending"):
                            m = re.search(r"([-+]?[0-9]*\.?[0-9]+)", total)
                            if m:
                                updates["total"] = float(m.group(1))
                        
                        log_receipt(f"Updates to apply: {updates}", doc_id_int)
                        
                        if not updates:
                            log_receipt("No changes detected, returning", doc_id_int)
                            return (
                                "No changes to save",
                                gr.update(choices=_get_pending_receipts_choices(), value=None),
                                {"needs_confirm": False, "pending_updates": None}
                            )
                        
                        # Check if this is a high-risk edit (total changed by >30%)
                        if "total" in updates:
                            doc = get_document_by_id(doc_id_int)
                            old_total = doc.get("total") if doc else None
                            new_total = updates["total"]
                            log_receipt(f"Total change detected: OLD=${old_total} → NEW=${new_total}", doc_id_int)
                            
                            if old_total and old_total != 0:
                                pct_change = abs((new_total - old_total) / old_total) * 100
                                log_receipt(f"Percentage change: {pct_change:.1f}%", doc_id_int)
                                
                                # High-risk threshold: >30% change
                                if pct_change > 30:
                                    log_receipt(f"⚠️ HIGH-RISK: Change exceeds 30% threshold", doc_id_int)
                                    # Check if this is first click (needs confirmation)
                                    if not confirm_data.get("needs_confirm"):
                                        log_receipt("First click - showing warning, awaiting confirmation", doc_id_int)
                                        # First click: show warning and set flag
                                        warning_msg = (
                                            f"⚠️ HIGH-RISK EDIT:\n"
                                            f"Total changing by {pct_change:.1f}%\n"
                                            f"Old: ${old_total:.2f} → New: ${new_total:.2f}\n"
                                            f"\n**Click Save again to confirm.**"
                                        )
                                        return (
                                            warning_msg,
                                            gr.update(choices=_get_pending_receipts_choices(), value=None),
                                            {"needs_confirm": True, "pending_updates": updates}
                                        )
                                    else:
                                        log_receipt("Second click - CONFIRMED high-risk change", doc_id_int)
                                        # Second click: proceed with update using stored updates
                                        updates = confirm_data.get("pending_updates", updates)
                        
                        # Apply auto-clear pending status
                        has_vendor = updates.get("vendor") or (vendor and vendor.strip())
                        has_date = updates.get("doc_date") or (date and date.strip())
                        has_total = updates.get("total") or (total and total > 0)
                        
                        log_receipt(f"Pending clear check: vendor={bool(has_vendor)}, date={bool(has_date)}, total={bool(has_total)}", doc_id_int)
                        
                        if has_vendor and has_date and has_total:
                            updates["is_pending"] = 0
                            log_receipt("Auto-clearing pending status (all required fields present)", doc_id_int)
                        
                        # Save to database
                        log_receipt("Executing database update with guardrails validation", doc_id_int)
                        update_document(doc_id_int, updates)
                        log_receipt("✅ Database update successful", doc_id_int)
                        
                        # Get integrity check result
                        log_receipt("Running post-save integrity check", doc_id_int)
                        check_result = integrity_check(doc_id_int)
                        log_receipt(f"Integrity check result: {check_result}", doc_id_int)
                        
                        status_text = f"✓ Saved doc #{doc_id_int}\n"
                        
                        if check_result.get("status") == "verified":
                            status_text += "📋 Status: **Verified** ✅"
                            log_receipt("Document status: VERIFIED ✅", doc_id_int)
                        elif check_result.get("status") == "pending":
                            missing = ", ".join(check_result.get("missing_fields", []))
                            status_text += f"📋 Status: **Pending** ⏳ (missing: {missing})"
                            log_receipt(f"Document status: PENDING ⏳ (missing: {missing})", doc_id_int)
                        else:
                            status_text += "📋 Status: **Has Mismatch Flag** ⚠️"
                            log_receipt("Document status: MISMATCH FLAG ⚠️", doc_id_int)
                        
                        if check_result.get("edit_count"):
                            status_text += f"\n📝 Edits: {check_result['edit_count']}"
                            log_receipt(f"Total edits recorded: {check_result['edit_count']}", doc_id_int)
                        
                        log_receipt(f"SAVE OPERATION COMPLETED SUCCESSFULLY\n{'='*60}\n", doc_id_int)
                        
                        return (
                            status_text,
                            gr.update(choices=_get_pending_receipts_choices(), value=None),
                            {"needs_confirm": False, "pending_updates": None}
                        )
                    
                    except ValueError as e:
                        # Validation error from guardrails - display exact message cleanly
                        error_msg = f"⚠ Validation error: {str(e)}"
                        log_receipt(f"❌ VALIDATION ERROR: {str(e)}", int(doc_id) if doc_id else None)
                        logger.debug(f"Validation error for doc #{doc_id}: {str(e)}")
                        return (
                            error_msg,
                            gr.update(choices=_get_pending_receipts_choices(), value=None),
                            {"needs_confirm": False, "pending_updates": None}
                        )
                    
                    except Exception as e:
                        # Unexpected error - show short message, log full details
                        error_msg = f"❌ Error saving document. Check console for details."
                        logger.error(
                            f"Unexpected error saving doc #{doc_id}:\n{traceback.format_exc()}",
                            exc_info=True
                        )
                        return (
                            error_msg,
                            gr.update(choices=_get_pending_receipts_choices(), value=None),
                            {"needs_confirm": False, "pending_updates": None}
                        )


                
                save_btn.click(
                    save_pending_changes,
                    [selected_doc_id, edit_vendor, edit_category, edit_date, edit_total, confirm_state],
                    [status_msg, pending_radio, confirm_state],
                )
                
                
                view_history_btn.click(
                    fn=render_edit_history,
                    inputs=[selected_doc_id],
                    outputs=[history_md],
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
