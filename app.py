import streamlit as st
from langchain_ollama import ChatOllama
from langchain.prompts import PromptTemplate
from langchain.retrievers.web_research import WebResearchRetriever
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.schema.runnable import RunnableLambda
import json
import os

st.set_page_config(layout="wide")
st.title("🔗 Cross-Prompt IA com Memória e Enriquecimento Web")

# Modelos locais via Ollama
llm_1 = ChatOllama(model="mistral", temperature=0.4)
llm_2 = ChatOllama(model="llama3", temperature=0.5)
llm_3 = ChatOllama(model="phi3", temperature=0.3)

# Vetorizador RAG
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.load_local("rag_index", embeddings=embedding_model) if os.path.exists("rag_index") else FAISS.from_texts([""], embedding_model)

# Web Retriever
def retrieve_web_context(query: str):
    retriever = WebResearchRetriever.from_llm_and_tools(
        llm=llm_1,
        search_sources=["duckduckgo"],
        vectorstore=vectorstore,
        k=3
    )
    documents = retriever.invoke({"question": query})
    return "\n".join([doc.page_content for doc in documents])

# PROMPTS
prompt_template_1 = PromptTemplate.from_template("""
Você é um assistente especializado em análise de tarefas complexas. [...]""")
chain_1 = prompt_template_1 | llm_1

prompt_template_2 = PromptTemplate.from_template("""
Com base na análise técnica a seguir [...]""")
chain_2 = prompt_template_2 | llm_2

prompt_template_3 = PromptTemplate.from_template("""
A seguir está um plano de ação técnico [...]""")
chain_3 = prompt_template_3 | llm_3

# Interface Streamlit
st.sidebar.header("Entrada")
tarefa = st.sidebar.text_area("Descreva a tarefa:", value="Desenvolver um sistema de orquestração e enriquecimento de prompt para LLMs com foco em uso laboratorial e automação analítica.")
executar = st.sidebar.button("Executar análise completa")
etapa_selecionada = st.sidebar.radio("Executar etapa específica:", ["Nenhuma", "Resumo Técnico", "Plano de Ação", "Avaliação Crítica"])
exportar = st.sidebar.button("Exportar Resultado em Markdown")

# Carregar memória
if os.path.exists("memoria_crossprompt.json"):
    with open("memoria_crossprompt.json", "r", encoding="utf-8") as f:
        memoria = json.load(f)
else:
    memoria = {}

st.sidebar.markdown("---")
st.sidebar.write("🧠 Memória atual:")
with st.sidebar.expander("Visualizar memória", expanded=False):
    for chave, valor in memoria.items():
        st.markdown(f"**{chave.upper()}**\n\n{valor[:500]}{'...' if len(valor) > 500 else ''}")

context = {"tarefa": tarefa}
web_context = ""
resumo = plano = avaliacao = ""

if executar or etapa_selecionada != "Nenhuma":
    with st.spinner("🔍 Buscando contexto na web..."):
        web_context = retrieve_web_context(tarefa)

if executar or etapa_selecionada == "Resumo Técnico":
    with st.spinner("✏️ Gerando resumo técnico..."):
        resumo = chain_1.invoke({"tarefa": tarefa, "memoria": json.dumps(memoria), "web_context": web_context}).content
        st.subheader("Resumo Técnico")
        st.markdown(resumo)
        context["resumo"] = resumo

if executar or etapa_selecionada == "Plano de Ação":
    if not resumo:
        resumo = memoria.get("resumo", "")
    with st.spinner("📘 Elaborando plano de ação..."):
        plano = chain_2.invoke({"resumo": resumo, "memoria": json.dumps(memoria), "web_context": web_context}).content
        st.subheader("Plano de Ação")
        st.markdown(plano)
        context["plano"] = plano

if executar or etapa_selecionada == "Avaliação Crítica":
    if not plano:
        plano = memoria.get("plano", "")
    with st.spinner("🔎 Realizando avaliação crítica..."):
        avaliacao = chain_3.invoke({"plano": plano, "memoria": json.dumps(memoria), "web_context": web_context}).content
        st.subheader("Avaliação Crítica")
        st.markdown(avaliacao)
        context["avaliacao"] = avaliacao

# Atualiza memória
if executar:
    memoria.update({"tarefa": tarefa, "web_context": web_context, "resumo": resumo, "plano": plano, "avaliacao": avaliacao})
    with open("memoria_crossprompt.json", "w", encoding="utf-8") as f:
        json.dump(memoria, f, ensure_ascii=False, indent=4)
    st.sidebar.success("🧠 Memória atualizada com sucesso!")

# Exportar em markdown
if exportar:
    with open("resultado_crossprompt.md", "w", encoding="utf-8") as f:
        f.write(f"# Tarefa\n{tarefa}\n\n")
        if "resumo" in context:
            f.write(f"## Resumo Técnico\n{context['resumo']}\n\n")
        if "plano" in context:
            f.write(f"## Plano de Ação\n{context['plano']}\n\n")
        if "avaliacao" in context:
            f.write(f"## Avaliação Crítica\n{context['avaliacao']}\n\n")
    st.success("✅ Resultado exportado como 'resultado_crossprompt.md'")
<substituiremos com o conteúdo do canvas em próxima célula>
