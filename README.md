# 🧠 Cross-Prompt IA com RAG e Memória

Este projeto implementa uma interface em Streamlit que encadeia múltiplos modelos LLM usando a estratégia Cross-Prompt com memória acumulada e enriquecimento de contexto via RAG (web search).

## Funcionalidades

- Cadeia de raciocínio entre 3 modelos diferentes via LangChain + Ollama
- Memória persistente entre execuções (JSON)
- Busca na web via DuckDuckGo
- Exportação do resultado em Markdown
- Interface interativa com Streamlit

## Executar localmente

```bash
pip install -r requirements.txt
streamlit run app.py
```

> Certifique-se de ter o [Ollama](https://ollama.com) instalado e os modelos `mistral`, `llama3` e `phi3` prontos.

## Deploy no Streamlit Cloud

1. Suba os arquivos para um repositório público no GitHub.
2. Acesse [https://streamlit.io/cloud](https://streamlit.io/cloud).
3. Conecte seu repositório e selecione `app.py`.
4. Clique em **Deploy** e pronto.
