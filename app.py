# app.py (Başlangıç)
import streamlit as st
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain_community.callbacks import get_openai_callback
import numpy as np

# --- 0. GENEL AYARLAR VE YÜKLEMELER ---
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_MODEL_NAME = "gpt-3.5-turbo"
EMBEDDINGS_MODEL_NAME = "text-embedding-3-small"

# --- 1. PDF İŞLEME FONKSİYONLARI ---
def process_pdfs(pdf_files):
    """
    Yüklenen PDF dosyalarını okur, metinlerini çıkarır ve LangChain Document nesnelerine dönüştürür.
    """
    all_docs_content = []
    for pdf_file in pdf_files:
        temp_file_path = os.path.join(".", pdf_file.name)
        with open(temp_file_path, "wb") as f:
            f.write(pdf_file.getbuffer())
        loader = PyMuPDFLoader(temp_file_path)
        documents = loader.load()
        all_docs_content.extend(documents)
        os.remove(temp_file_path)
    return all_docs_content

def split_text_into_chunks(documents):
    """
    LangChain Document listesini anlamlı parçalara (chunks) böler.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=900,
        chunk_overlap=150,
        length_function=len
    )
    chunks = text_splitter.split_documents(documents)
    return chunks

# --- 2. EMBEDDING TABANLI CHUNK SEÇİMİ ---
@st.cache_resource
def get_embeddings_model():
    return OpenAIEmbeddings(model=EMBEDDINGS_MODEL_NAME, openai_api_key=OPENAI_API_KEY)

def embed_chunks(chunks, embeddings_model):
    texts = [chunk.page_content for chunk in chunks]
    vectors = embeddings_model.embed_documents(texts)
    return np.array(vectors)

def embed_query(query, embeddings_model):
    return np.array(embeddings_model.embed_query(query))

def find_relevant_chunks(query, chunks, embeddings_model, chunk_vectors, k=5):
    if not chunks or chunk_vectors is None:
        return []
    query_vec = embed_query(query, embeddings_model)
    sims = np.dot(chunk_vectors, query_vec) / (np.linalg.norm(chunk_vectors, axis=1) * np.linalg.norm(query_vec) + 1e-8)
    top_indices = np.argsort(sims)[-k:][::-1]
    return [chunks[i] for i in top_indices]

# --- 3. SORU CEVAPLAMA MEKANİZMASI ---
def get_llm():
    return ChatOpenAI(model_name=LLM_MODEL_NAME, temperature=0.2, openai_api_key=OPENAI_API_KEY)

def generate_answer_with_llm(query, relevant_chunks, llm):
    if not relevant_chunks:
        return "Üzgünüm, bu soruya cevap verecek ilgili bilgi ders notlarınızda bulunamadı."
    prompt_template_text = """
Sen bir Computer Engineering alanında uzman bir asistan ve sınav koçusun. Kullanıcıdan gelen SORU'yu, sadece aşağıdaki BAĞLAMdaki bilgilere dayanarak, vize/final sınavına hazırlanan bir öğrenciye veya konuyu hızlıca öğrenmek isteyen birine yönelik, kısa, öz, teknik ve anlaşılır bir dille yanıtla.

- Gereksiz detay, referans veya akademik dil kullanma.
- Sadece özet, anahtar noktalar, formüller, temel kavramlar ve sınavda işine yarayacak püf noktalarını vurgula.
- Gerekirse örnek, kısa açıklama veya formül ekle.
- Cevabın Computer Engineering alanında profesyonel ve pratik olsun.

BAĞLAM:
{context}

SORU: {question}

CEVAP (Kısa, öz, teknik, sınav odaklı):"""
    CUSTOM_PROMPT = PromptTemplate(
        template=prompt_template_text, input_variables=["context", "question"]
    )
    chain = load_qa_chain(llm, chain_type="stuff", prompt=CUSTOM_PROMPT)
    with get_openai_callback() as cb:
        try:
            response = chain.invoke({"input_documents": relevant_chunks, "question": query})
            answer = response.get('output_text', "Cevap alınırken bir sorun oluştu veya 'output_text' anahtarı bulunamadı.")
            st.session_state.token_usage = {
                "total_tokens": cb.total_tokens,
                "prompt_tokens": cb.prompt_tokens,
                "completion_tokens": cb.completion_tokens,
                "total_cost_usd": cb.total_cost
            }
        except Exception as e:
            st.error(f"LLM'den cevap alınırken hata: {e}")
            return "Cevap üretilirken bir sorun oluştu."
    return answer

def generate_summary(chunks, llm, summary_type="kısa"):
    context = "\n".join([chunk.page_content for chunk in chunks[:8]])  # ilk 8 chunk (yaklaşık ilk 6-8 sayfa)
    prompt = f"""
Aşağıda Computer Engineering alanında bir ders notundan alınmış metinler var. Bu metinleri {summary_type} ve sınav/pratik odaklı, teknik bir özet haline getir. Gereksiz detay verme, anahtar noktaları vurgula.

METİN:
{context}

ÖZET ({summary_type}):
"""
    with get_openai_callback() as cb:
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            st.error(f"Özet üretilirken hata: {e}")
            return "Özet üretilemedi."

def generate_flashcards(chunks, llm):
    context = "\n".join([chunk.page_content for chunk in chunks[:10]])
    prompt = f"""
Aşağıda Computer Engineering alanında bir ders notundan alınmış metinler var. Bu metinlerden sınav/pratik odaklı, teknik 5 adet flashcard (soru-cevap kartı) üret. Her kartta kısa bir soru ve kısa bir teknik cevap olsun. Sadece kartları listele.

METİN:
{context}

FLASHCARD LİSTESİ:
"""
    with get_openai_callback() as cb:
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            st.error(f"Flashcard üretilirken hata: {e}")
            return "Flashcard üretilemedi."

def generate_quiz(chunks, llm):
    context = "\n".join([chunk.page_content for chunk in chunks[:10]])
    prompt = f"""
Aşağıda Computer Engineering alanında bir ders notundan alınmış metinler var. Bu metinlerden sınav/pratik odaklı, teknik 3 adet çoktan seçmeli quiz sorusu üret. Her sorunun 1 doğru, 3 yanlış şıkkı olsun. Sadece quiz sorularını ve şıklarını listele.

METİN:
{context}

QUIZ SORULARI:
"""
    with get_openai_callback() as cb:
        try:
            response = llm.invoke(prompt)
            return response.content
        except Exception as e:
            st.error(f"Quiz üretilirken hata: {e}")
            return "Quiz üretilemedi."

# --- DÖKÜMAN İÇİ ARAMA VE SAYFA ATLAMA ---
def search_in_document(query, chunks):
    results = []
    for i, chunk in enumerate(chunks):
        if query.lower() in chunk.page_content.lower():
            page = chunk.metadata.get('page', i+1) if hasattr(chunk, 'metadata') else i+1
            results.append({
                'chunk_index': i,
                'page': page,
                'content': chunk.page_content
            })
    return results

# --- MODERN ARAYÜZ VE TEMA ---
def set_modern_theme():
    st.markdown(
        """
        <style>
        .stApp {background-color: #f7f9fa;}
        .block-container {padding-top: 2rem;}
        .stButton>button {background-color: #0056b3; color: white; border-radius: 6px; font-weight: 600;}
        .stButton>button:hover {background-color: #003d80;}
        .stTextInput>div>input {border-radius: 6px; border: 1.5px solid #0056b3;}
        .stSidebar {background-color: #e9ecef;}
        .stInfo, .stSuccess, .stWarning {border-radius: 8px;}
        .stChatMessage {border-radius: 8px;}
        </style>
        """,
        unsafe_allow_html=True
    )

# --- 4. STREAMLIT KULLANICI ARAYÜZÜ ---
def main():
    st.set_page_config(page_title="Akademik Soru-Cevap Chatbotu", page_icon="💻", layout="wide")
    st.header("💻 Computer Engineering Sınav Koçu ve Akıllı PDF Chatbotu")
    st.markdown("""
    Bu chatbot, yüklediğiniz PDF ders notlarınız üzerinden **Computer Engineering** alanında, vize/final ve pratik öğrenme odaklı, profesyonel ve hızlı cevaplar vermek için tasarlanmıştır.
    PDF'inizi yükleyin, ardından teknik, sınav odaklı ve anlaşılır cevaplar alın.
    """)

    if not OPENAI_API_KEY:
        st.error("Lütfen .env dosyasında OPENAI_API_KEY değişkenini ayarlayın!")
        return

    set_modern_theme()

    with st.sidebar:
        st.subheader("📄 Ders Notlarınızı Yükleyin")
        uploaded_files = st.file_uploader(
            "PDF dosyalarınızı buraya sürükleyip bırakın veya seçin",
            type="pdf",
            accept_multiple_files=True,
            key="pdf_uploader"
        )
        if uploaded_files:
            st.success(f"{len(uploaded_files)} PDF yüklendi. Şimdi soru sorabilirsiniz.")
            st.markdown("---")
            if st.button("Otomatik Özet Üret", key="summary_btn"):
                with st.spinner("Özet üretiliyor..."):
                    raw_documents = process_pdfs(uploaded_files)
                    chunks = split_text_into_chunks(raw_documents)
                    llm = get_llm()
                    summary = generate_summary(chunks, llm, summary_type="kısa")
                    st.session_state.generated_summary = summary
            if st.button("Flashcard Üret", key="flashcard_btn"):
                with st.spinner("Flashcard üretiliyor..."):
                    raw_documents = process_pdfs(uploaded_files)
                    chunks = split_text_into_chunks(raw_documents)
                    llm = get_llm()
                    flashcards = generate_flashcards(chunks, llm)
                    st.session_state.generated_flashcards = flashcards
            if st.button("Quiz Üret", key="quiz_btn"):
                with st.spinner("Quiz üretiliyor..."):
                    raw_documents = process_pdfs(uploaded_files)
                    chunks = split_text_into_chunks(raw_documents)
                    llm = get_llm()
                    quiz = generate_quiz(chunks, llm)
                    st.session_state.generated_quiz = quiz
            st.markdown("---")
            st.subheader("🔍 Döküman İçi Arama")
            search_query = st.text_input("Anahtar kelime ile ara...", key="doc_search")
            if uploaded_files and search_query:
                raw_documents = process_pdfs(uploaded_files)
                chunks = split_text_into_chunks(raw_documents)
                search_results = search_in_document(search_query, chunks)
                if search_results:
                    st.write(f"{len(search_results)} sonuç bulundu:")
                    for res in search_results:
                        if st.button(f"Sayfa {res['page']} - Sonucu Göster", key=f"show_{res['chunk_index']}"):
                            st.session_state.selected_search_result = res
                else:
                    st.info("Aradığınız kelimeyle eşleşen içerik bulunamadı.")
            st.markdown("---")
        st.markdown("---")
        if "token_usage" in st.session_state and st.session_state.token_usage:
            usage = st.session_state.token_usage
            st.subheader("Son Sorgu API Kullanımı:")
            st.text(f"Toplam Token: {usage['total_tokens']}")
            st.text(f"İstem Tokenları: {usage['prompt_tokens']}")
            st.text(f"Cevap Tokenları: {usage['completion_tokens']}")
            st.text(f"Tahmini Maliyet: ${usage['total_cost_usd']:.6f}")
            st.session_state.token_usage = None

    if "messages" not in st.session_state:
        st.session_state.messages = [{"role": "assistant", "content": "Merhaba! Computer Engineering alanında, PDF'inizden sınav ve pratik odaklı teknik sorular sorabilirsiniz. Lütfen önce soldaki menüden PDF'lerinizi yükleyin."}]

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    query = st.chat_input("Sorunuzu buraya yazın...", key="chat_input")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            if not uploaded_files:
                full_response = "Lütfen önce sol taraftan PDF dosyalarınızı yükleyin."
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
                st.stop()
            with st.spinner("Cevap aranıyor ve oluşturuluyor..."):
                try:
                    raw_documents = process_pdfs(uploaded_files)
                    chunks = split_text_into_chunks(raw_documents)
                    embeddings_model = get_embeddings_model()
                    chunk_vectors = embed_chunks(chunks, embeddings_model)
                    relevant_chunks = find_relevant_chunks(query, chunks, embeddings_model, chunk_vectors)
                    llm = get_llm()
                    full_response = generate_answer_with_llm(query, relevant_chunks, llm)
                    with st.expander("📄 Cevap için Kullanılan Kaynaklar (Ders Notu Kesitleri)"):
                        for i, chunk in enumerate(relevant_chunks):
                            st.markdown(f"**Kaynak {i+1}:**")
                            st.caption(chunk.page_content[:500] + "..." if len(chunk.page_content) > 500 else chunk.page_content)
                            if hasattr(chunk, 'metadata') and chunk.metadata:
                                st.json(chunk.metadata, expanded=False)
                except Exception as e:
                    st.error(f"Cevap alınırken bir hata oluştu: {e}")
                    full_response = "Üzgünüm, bir sorunla karşılaştım ve cevap üretemedim."
            message_placeholder.markdown(full_response)
        st.session_state.messages.append({"role": "assistant", "content": full_response})

    if "generated_summary" in st.session_state:
        st.info("### Otomatik Özet:\n" + st.session_state.generated_summary)
    if "generated_flashcards" in st.session_state:
        st.success("### Flashcard Kartları:\n" + st.session_state.generated_flashcards)
    if "generated_quiz" in st.session_state:
        st.warning("### Quiz Soruları:\n" + st.session_state.generated_quiz)

    if "selected_search_result" in st.session_state:
        res = st.session_state.selected_search_result
        st.info(f"**Sayfa {res['page']} - Arama Sonucu:**\n" + res['content'])

if __name__ == "__main__":
    main()