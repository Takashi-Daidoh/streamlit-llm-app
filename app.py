# =========================
# config
# =========================
import os
import re
from dataclasses import dataclass
from typing import Optional, List

from dotenv import load_dotenv
load_dotenv()  # ローカル .env 読み込み

import streamlit as st
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import SystemMessage, HumanMessage, Document  # Documentはフォールバック用に使用


@dataclass
class Config:
    app_name: str = "streamlit-llm-app"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    embed_model: str = "text-embedding-3-small"
    top_k: int = 3


# =========================
# utils
# =========================
def slugify_ascii(text: str) -> str:
    """ASCII/slug化（パス安全化）"""
    text = text.lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    return re.sub(r"-+", "-", text).strip("-")


def build_llm(cfg: Config) -> ChatOpenAI:
    """LLMクライアント生成"""
    return ChatOpenAI(model_name=cfg.model_name, temperature=cfg.temperature)


def resolve_api_key_or_error() -> Optional[str]:
    """ENV→Secretsの順でAPIキーを取得し、見つかればENVにも注入"""
    key = os.getenv("OPENAI_API_KEY")
    if key:
        return key
    try:
        key = st.secrets.get("OPENAI_API_KEY")  # SecretsのTOML不正だと例外
    except Exception:
        st.error(
            "Streamlit Secrets の構文（TOML）が不正です。以下の形式に直してください：\n\n"
            '```\nOPENAI_API_KEY = "sk-xxxxxxxx"\n```'
        )
        return None
    if key:
        os.environ["OPENAI_API_KEY"] = key
        return key
    return None


# =========================
# stores
# =========================
class SimpleInMemoryRetriever:
    """Chromaが使えない環境向けの簡易ベクタ検索（OpenAI埋め込み + コサイン類似度）"""
    def __init__(self, texts: List[str], metadatas: List[dict], embeddings: OpenAIEmbeddings, k: int = 3):
        self.embeddings = embeddings
        self.docs = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
        self.vectors = embeddings.embed_documents(texts)
        self.k = k

    def _cosine(self, u: List[float], v: List[float]) -> float:
        # numpyなしのコサイン類似度
        dot = sum(a * b for a, b in zip(u, v))
        nu = sum(a * a for a in u) ** 0.5
        nv = sum(b * b for b in v) ** 0.5
        return dot / (nu * nv + 1e-10)

    def get_relevant_documents(self, query: str) -> List[Document]:
        qv = self.embeddings.embed_query(query)
        scored = [(self._cosine(qv, vec), i) for i, vec in enumerate(self.vectors)]
        scored.sort(reverse=True)
        top = scored[: self.k]
        return [self.docs[i] for _, i in top]


def build_retriever(cfg: Config):
    """可能ならChroma(エフェメラル)を使用。ダメならインメモリ検索にフォールバック。"""
    texts = [
        "persona:A 方針: あなたは日本の労務・就業規則の一般的な助言者。一次情報の確認を促し、個別の法的助言は避け、実務チェックリストや相談先を提案する。",
        "persona:B 方針: あなたはPython/生成AIの業務改善エンジニア。小さく作って検証、再現性とセキュリティ、APIキー管理に配慮し、簡潔な手順を示す。",
    ]
    metadatas = [{"persona": "A"}, {"persona": "B"}]
    embeddings = OpenAIEmbeddings(model=cfg.embed_model)

    # --- まずChromaを試す（import段階で落ちる可能性があるため関数内でtry） ---
    try:
        import chromadb
        from chromadb.config import Settings
        from langchain_community.vectorstores import Chroma

        client = chromadb.EphemeralClient(Settings(is_persistent=False, anonymized_telemetry=False))
        vs = Chroma.from_texts(
            texts=texts,
            embedding=embeddings,
            metadatas=metadatas,
            client=client,
            collection_name=f"persona-{slugify_ascii(cfg.app_name)}",
        )
        return vs.as_retriever(search_kwargs={"k": cfg.top_k})
    except Exception as e:
        # CloudでSQLiteバージョンが古い等の理由で失敗 → フォールバック
        st.warning(f"Chromaを利用できないため、インメモリ検索にフォールバックします: {e}")
        return SimpleInMemoryRetriever(texts, metadatas, embeddings, k=cfg.top_k)


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
            return "OpenAI APIキーの問題が発生しました。.env または Secrets を確認してください。"
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
            "- ローカルは `.env`、Cloudは **Secrets** に `OPENAI_API_KEY` を設定してください。"
        )

    # APIキー解決（ENV or Secrets）
    api_key = resolve_api_key_or_error()
    if not api_key:
        st.stop()

    cfg = Config()
    try:
        llm = build_llm(cfg)
    except Exception as e:
        st.error(f"LLM初期化でエラー: {e}")
        st.stop()

    retriever = None
    try:
        retriever = build_retriever(cfg)  # Chroma or インメモリに自動切替
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
    run_app()
