# =========================
# config
# =========================
# 環境・基本設定（.env から OPENAI_API_KEY を読み込む）
import os
import re
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv  # .env 読み込み
load_dotenv()

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage
from langchain_community.vectorstores import Chroma


# 基本設定をひとまとめにする
@dataclass
class Config:
    app_name: str = "streamlit-llm-app"            # アプリ名（DBパスのスラグ化に使用）
    model_name: str = "gpt-4o-mini"                # LLMモデル
    temperature: float = 0.0                       # 出力の多様性
    embed_model: str = "text-embedding-3-small"    # 埋め込みモデル
    top_k: int = 3                                 # RAGの取得件数


# =========================
# utils
# =========================
def slugify_ascii(text: str) -> str:
    """ASCII/slug化（ファイルパス用に安全化）"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def build_llm(cfg: Config) -> ChatOpenAI:
    """LLMクライアントを生成（依存注入のため分離）"""
    return ChatOpenAI(model_name=cfg.model_name, temperature=cfg.temperature)


# =========================
# stores
# =========================
def build_vectorstore(cfg: Config) -> Chroma:
    """Chroma(0.4+)で軽量の方針テキストを格納（persist()不要）"""
    # persona方針（最小の知識ベース）
    texts = [
        "persona:A 方針: あなたは日本の労務・就業規則の一般的な助言者。法令の最終判断は一次情報や専門家確認を促す。個別の法的助言は避け、実務チェックリストや相談先を提案する。",
        "persona:B 方針: あなたはPython/生成AIの業務改善エンジニア。小さく作って検証、ログと再現性を重視し、APIキー管理とセキュリティに配慮した手順を簡潔に示す。",
    ]
    metadatas = [{"persona": "A"}, {"persona": "B"}]

    embeddings = OpenAIEmbeddings(model=cfg.embed_model)
    persist_dir = f".chroma-{slugify_ascii(cfg.app_name)}"  # DBパスはASCII/slug化
    # ここでは簡便に毎回 from_texts で作成（課題要件上OK）
    vs = Chroma.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas, persist_directory=persist_dir)
    return vs


def build_retriever(cfg: Config):
    """VectorStoreからRetrieverを生成"""
    vs = build_vectorstore(cfg)
    return vs.as_retriever(search_kwargs={"k": cfg.top_k})


# =========================
# indexing
# =========================
def retrieve_context(query: str, persona: str, retriever) -> List[str]:
    """クエリ＋ペルソナで関連文脈を取得（失敗時は空リスト）"""
    try:
        docs = retriever.get_relevant_documents(query)
        selected = [d.page_content for d in docs if d.metadata.get("persona") == persona]
        return selected or [d.page_content for d in docs]
    except Exception:
        return []


# =========================
# tools
# =========================
def make_system_prompt(persona: str, contexts: Optional[List[str]] = None) -> str:
    """ペルソナ別のSystemメッセージを生成し、補助文脈を付加"""
    persona_map = {
        "A": (
            "You are a Japanese labor/HR compliance advisor. "
            "Provide general, practical, and safe guidance. "
            "For legal decisions, encourage users to check official sources or consult professionals."
        ),
        "B": (
            "You are a Python & Generative AI productivity engineer. "
            "Propose small testable steps, concise examples, and consider security & reproducibility."
        ),
    }
    base = persona_map.get(persona, "You are a helpful assistant.")
    ctx = ""
    if contexts:
        ctx = "\n\n# Additional context (persona policy)\n" + "\n".join(f"- {c}" for c in contexts)
    return base + ctx


# =========================
# agent
# =========================
def generate_response(
    user_text: str,
    persona_value: str,
    llm: ChatOpenAI,
    retriever=None,
) -> str:
    """入力テキストとラジオ選択を受け取り、LLMの回答文字列を返す"""
    try:
        if not user_text.strip():
            return "入力が空です。相談内容を入力してください。"

        persona = "A" if persona_value.startswith("A") else "B"
        contexts = retrieve_context(user_text, persona, retriever) if retriever else []
        system_prompt = make_system_prompt(persona, contexts)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_text),
        ]
        result = llm.invoke(messages)  # LangChain v0.2+ の推奨呼び出し
        return result.content or "（LLMから空の応答が返りました）"

    except Exception as e:
        msg = str(e)
        if "OPENAI_API_KEY" in msg or "api key" in msg.lower():
            return "OpenAI APIキーの問題が発生しました。.env の OPENAI_API_KEY を確認してください。"
        if "rate limit" in msg.lower():
            return "レート制限に達しました。しばらく待ってから再実行してください。"
        if "timeout" in msg.lower():
            return "ネットワーク遅延またはタイムアウトが発生しました。接続環境を確認してください。"
        return f"エラーが発生しました。原因の推測: {msg}"


# =========================
# app (Streamlit)
# =========================
def run_app():
    """Streamlit UI 本体"""
    st.set_page_config(page_title="LLMアプリ（LangChain + Streamlit）", page_icon="🤖", layout="centered")

    st.title("LLMアプリ（LangChain + Streamlit）")
    st.caption("入力フォームに相談内容を記入し、専門家A/Bを選んで送信してください。LangChain経由でLLMが回答します。")

    with st.expander("このアプリの概要 / 操作方法", expanded=True):
        st.markdown(
            "- 入力フォームは1つです。\n"
            "- ラジオで **専門家タイプ** を選択（A: 労務アドバイザー / B: 業務改善エンジニア）。\n"
            "- 選択に応じて System メッセージを切り替え、LLMに投げます。\n"
            "- APIキーは `.env` の `OPENAI_API_KEY` を使用します（GitHubへは含めない）。"
        )

    if not os.getenv("OPENAI_API_KEY"):
        st.error("OPENAI_API_KEY が環境変数に見つかりません。.env を確認してください。")
        st.stop()

    cfg = Config()
    try:
        llm = build_llm(cfg)  # LLM初期化
    except Exception as e:
        st.error(f"LLM初期化でエラー: {e}")
        st.stop()

    retriever = None
    try:
        retriever = build_retriever(cfg)  # 軽量RAG
    except Exception as e:
        st.warning(f"知識ベース初期化に失敗しました（RAGなしで継続）: {e}")

    persona = st.radio(
        "専門家の種類を選択",
        options=["A｜労務アドバイザー", "B｜業務改善エンジニア"],
        horizontal=True,
    )
    user_text = st.text_area("相談内容（入力フォーム）", height=160, placeholder="例：36協定の運用ポイント / 小さな自動化の始め方")
    if st.button("送信"):
        with st.spinner("LLMに問い合わせ中..."):
            answer = generate_response(user_text, persona, llm, retriever)
        st.markdown("### 回答")
        st.write(answer)


# =========================
# main（Streamlitエントリポイント）
# =========================
if __name__ == "__main__":
    run_app()  # Streamlitで起動した際にUIを描画
