---
title: Qwen2 VL LaTeX OCR MCP
emoji: 🧮
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.34.0
app_file: app.py
pinned: false
license: mit
tags:
  - mcp-server-track
---

# Qwen2-VL LaTeX OCR (MCP Server)

Converte imagens de equações matemáticas em código LaTeX usando o **Qwen2-VL-7B-Instruct**, rodando em **ZeroGPU**.

Este Space funciona de duas formas:

1. **UI Gradio**: upload de imagem, retorna o LaTeX.
2. **MCP Server**: exposto em `https://<seu-usuario>-<nome-do-space>.hf.space/gradio_api/mcp/sse`, com a tool `image_to_latex` disponível pra qualquer client MCP (Claude Desktop, Cursor, etc).

## Uso via MCP

Adicione ao seu client MCP:

\`\`\`json
{
  "mcpServers": {
    "latex-ocr": {
      "url": "https://<seu-usuario>-<nome-do-space>.hf.space/gradio_api/mcp/sse"
    }
  }
}
\`\`\`

## Stack

- Qwen2-VL-7B-Instruct (bf16)
- Gradio 5 + ZeroGPU
- MCP nativo (sem bridge externo)