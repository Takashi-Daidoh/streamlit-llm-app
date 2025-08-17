# =========================
# config: 環境変数の読み込み
# =========================
import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# .envから環境変数を読み込む
load_dotenv()


# =========================
# utils: 汎用関数
# =========================
def error_message(msg: str) -> str:
    # エラーメッセージを返す関数
    return f"⚠️ エラー: {msg}"


# =========================
# stores: DB関連（今回は未使用）
# =========================
def init_store():
    # 今回はベクターストアを使わないのでダミー
    return None


# =========================
# indexing: インデックス処理（未使用）
# =========================
def build_index():
    # 今回は未使用
    return None


# =========================
# tools: LLM呼び出し用
# =========================
def call_llm(user_input: str, expert: str, llm: ChatOpenAI) -> str:
    # 専門家ごとにsystemメッセージを切り替える
    if expert == "心理カウンセラー":
        system_prompt = "You are a professional psychological counselor. Provide empathetic and supportive advice."
    elif expert == "経営コンサルタント":
        system_prompt = "You are a skilled business consultant. Provide practical and strategic advice for management issues."
    else:
        return error_message("選択した専門家モードが不明です。")

    try:
        # LangChainのChatモデルにプロンプトを渡す
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_input),
        ]
        result = llm(messages)
        return result.content
    except Exception as e:
        return error_message(f"LLM呼び出しに失敗しました: {e}")


# =========================
# agent: 入力値を受けて処理
# =========================
def agent_process(user_input: str, expert: str) -> str:
    # LLMのインスタンスを生成
    try:
        llm = ChatOpenAI(model_name="gpt-4o-mini", temperature=0)
    except Exception as e:
        return error_message(f"LLM初期化に失敗しました: {e}")

    return call_llm(user_input, expert, llm)


# =========================
# app: Streamlit UI構築
# =========================
def main():
    st.title("🧠 LLMアプリ: 専門家に相談しよう")

    # アプリの説明文
    st.write("このアプリでは、入力した相談内容をLLMに渡し、")
    st.write("心理カウンセラー または 経営コンサルタント として回答します。")
    st.write("下記フォームに入力してください。")

    # ラジオボタン（専門家の選択）
    expert = st.radio("専門家を選んでください:", ["心理カウンセラー", "経営コンサルタント"])

    # 入力フォーム
    user_input = st.text_area("相談内容を入力してください:")

    # 実行ボタン
    if st.button("送信"):
        if not user_input.strip():
            st.warning("相談内容を入力してください。")
        else:
            with st.spinner("LLMが考えています..."):
                response = agent_process(user_input, expert)
                st.success("回答:")
                st.write(response)


# =========================
# 実行例
# =========================
if __name__ == "__main__":
    main()
