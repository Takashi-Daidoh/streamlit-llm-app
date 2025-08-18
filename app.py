# =========================
# SECTION: config
# =========================
import os  # 環境変数の参照に使用
from dotenv import load_dotenv  # .env を読み込む

# .env を読み込んで環境変数（OPENAI_API_KEYなど）を反映
load_dotenv()

def get_env_or_error(name: str) -> str:
    """必須の環境変数を取得。未設定なら分かりやすいエラー文を返す"""
    val = os.environ.get(name)
    if not val:
        raise RuntimeError(
            f"環境変数 '{name}' が見つかりません。.env の設定 or Streamlit CloudのSecretsを確認してください。"
        )
    return val


# =========================
# SECTION: utils
# =========================
import re
import unicodedata

def slugify(text: str) -> str:
    """DB/ファイル用に安全なASCIIスラッグへ変換"""
    text = unicodedata.normalize("NFKD", text)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r"[^a-zA-Z0-9._-]+", "-", text).strip("-._")
    text = re.sub(r"[-_.]+", "-", text)
    return text.lower() or "default"

def friendly_error(e: Exception) -> str:
    """ユーザー向けに原因が推測しやすいメッセージに整形"""
    msg = str(e)
    if "OPENAI_API_KEY" in msg:
        return "OpenAIのAPIキーが未設定です。.env もしくはStreamlitのSecretsで OPENAI_API_KEY を設定してください。"
    if "rate limit" in msg.lower():
        return "APIのレート制限に達しました。時間をおいて再実行してください。"
    if "network" in msg.lower():
        return "ネットワークエラーが発生しました。通信環境を確認してください。"
    return f"エラーが発生しました：{msg}"


# =========================
# SECTION: stores
# =========================
# ＊今回の課題ではRAGは不要ですが、将来拡張用の雛形だけ用意
from typing import Optional

try:
    from chromadb import Client as ChromaClient
    from chromadb.config import Settings as ChromaSettings
except Exception:
    ChromaClient = None  # chroma未インストールでも動くように回避
    ChromaSettings = None

def get_chroma(db_dir: str) -> Optional[object]:
    """Chroma 0.4+ を初期化（未使用だが雛形）。DBパスはASCII/slug化"""
    if ChromaClient is None:
        return None
    safe = slugify(db_dir)
    try:
        client = ChromaClient(settings=ChromaSettings(is_persistent=True, persist_directory=safe))
        return client
    except Exception as e:
        # 利用側で None チェックする想定
        print("[stores] Chroma初期化エラー:", e)
        return None


# =========================
# SECTION: indexing
# =========================
def build_index(docs: list, db_dir: str) -> str:
    """（雛形）将来ここでベクトル化・登録を行う"""
    # 今回は使わないためメッセージのみ返す
    return f"index skipped (docs={len(docs)}, db_dir={slugify(db_dir)})"


# =========================
# SECTION: tools
# =========================
# LangChainのChatOpenAIでLLMを呼び出す
from langchain_openai import ChatOpenAI  # Lesson8の書式に合わせる
from langchain.schema import SystemMessage, HumanMessage  # Chatメッセージの型

def get_llm(model: str = "gpt-4o-mini", temperature: float = 0.2) -> ChatOpenAI:
    """LLMインスタンスを生成（依存は引数注入の基本方針に従い、ここで作る/渡すも可）"""
    # APIキー未設定の場合はここで例外
    get_env_or_error("OPENAI_API_KEY")
    # Lesson8の記法に合わせ model_name を使用（環境によっては model に変更が必要）
    try:
        return ChatOpenAI(model_name=model, temperature=temperature)
    except TypeError:
        # ライブラリ差異のフォールバック
        return ChatOpenAI(model=model, temperature=temperature)

def system_prompt_for(expert: str) -> str:
    """ラジオボタンの選択に応じてSystemメッセージを切替"""
    if expert == "心理カウンセラー":
        return "You are a professional psychological counselor. Provide empathetic, non-judgmental, and supportive advice."
    if expert == "経営コンサルタント":
        return "You are a skilled business consultant. Provide practical, strategic, and actionable advice for management issues."
    # デフォルトの安全策
    return "You are a helpful assistant. Provide concise and helpful answers."


# =========================
# SECTION: agent
# =========================
def generate_answer(llm: ChatOpenAI, expert: str, user_text: str) -> str:
    """入力テキストと専門家種別から回答を生成して返す（失敗時は理由を含む）"""
    if not user_text or not user_text.strip():
        return "入力が空です。相談内容をテキストエリアに入力してください。"
    try:
        sys_msg = system_prompt_for(expert)
        messages = [
            SystemMessage(content=sys_msg),             # ロールの指示
            HumanMessage(content=user_text.strip()),    # ユーザ入力
        ]
        result = llm(messages)                          # LLM呼び出し
        return (result.content or "").strip() or "回答が空でした。プロンプトを見直してください。"
    except Exception as e:
        return friendly_error(e)


# =========================
# SECTION: app (Streamlit UI)
# =========================
import streamlit as st

def run_app() -> None:
    """Streamlitアプリ（入力フォーム1つ + ラジオ切替 + 回答表示）"""
    st.set_page_config(page_title="LLMアプリ（心理/経営）", page_icon="🧠", layout="centered")

    st.title("🧠 LLMアプリ（心理カウンセラー / 経営コンサルタント）")
    st.write(
        "- このアプリは LangChain + OpenAI を利用して回答を生成します。\n"
        "- 下のテキストに相談内容を入力し、**専門家の種類**を選んで送信してください。\n"
        "- APIキーは `.env` または Streamlit Community Cloud の Secrets に `OPENAI_API_KEY` で設定してください。"
    )

    # 専門家モードの選択
    expert = st.radio("専門家の種類を選択してください", ["心理カウンセラー", "経営コンサルタント"], horizontal=True)

    # 入力フォーム（1つ）
    user_text = st.text_area("相談内容（自由記述）", height=160, placeholder="例）最近、不安で眠れません。どうしたら良いでしょうか？")

    # 送信ボタン
    if st.button("送信", type="primary"):
        with st.spinner("回答を生成中..."):
            try:
                llm = get_llm()  # 依存注入：必要なら外から渡す設計にも対応可
                answer = generate_answer(llm, expert, user_text)
                st.markdown("### 回答")
                st.write(answer)
            except Exception as e:
                st.error(friendly_error(e))

    # フッターの簡易ヘルプ
    with st.expander("ℹ️ ヘルプ / トラブルシュート"):
        st.write(
            "- **エラー: APIキー未設定** → `.env` に `OPENAI_API_KEY=...` を記載（GitHubにはpushしない）。\n"
            "- **ModuleNotFoundError** → `pip install -r requirements.txt` を実行。\n"
            "- **Pythonバージョン** → 3.11 を使用（Cloudでは設定画面で 3.11 を選択）。"
        )

# Streamlit以外（python app.py直実行）でも最小限の動作例を用意
if __name__ == "__main__":
    # 簡易デモ（環境変数 RUN_DEMO=1 の時だけAPI呼び出し）
    if os.environ.get("RUN_DEMO") == "1":
        llm_demo = get_llm()
        demo = generate_answer(llm_demo, "経営コンサルタント", "SaaSの解約率を下げたい。すぐ効く対策を3つだけ。")
        print("=== Demo ===")
        print(demo)
    else:
        print("Streamlitアプリを起動するには:  `streamlit run app.py`")
