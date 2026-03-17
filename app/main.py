"""Application entrypoint."""

from __future__ import annotations

from typing import Optional

import gradio as gr

from app.agent import handle_message


def _format_assistant_response(payload: dict) -> str:
    response = str(payload.get("response", ""))
    citations = payload.get("citations", []) or []
    if citations:
        response += "\n\nCitations: " + ", ".join(str(c) for c in citations)
    return response


def _on_submit(
    message: str,
    file_path: Optional[str],
    history: list[tuple[str, str]],
) -> tuple[str, None, list[tuple[str, str]]]:
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
    history.append((user_text, assistant_text))
    return "", None, history


def build_app() -> gr.Blocks:
    with gr.Blocks(title="ReceiptIQ") as demo:
        gr.Markdown("# ReceiptIQ")
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

    return demo


def main() -> None:
    app = build_app()
    app.launch(share=True)


if __name__ == "__main__":
    main()
