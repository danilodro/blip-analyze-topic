import spacy
from collections import Counter, defaultdict
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from textblob import TextBlob
from spacy.lang.pt.stop_words import STOP_WORDS
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from gensim import corpora, models

app = FastAPI()

nlp = spacy.load("pt_core_news_sm")

class ConversationInput(BaseModel):
    messages: List[str]

nltk.download("stopwords")
nltk.download("punkt")

stop_words_lower = list(STOP_WORDS)

vectorizer = CountVectorizer(stop_words=stop_words_lower)

# palavras irrelevantes p remoção
irrelevant_keywords = ["cliente", "atendente", "senhor,", "senhora", "ok"] 

def get_main_topic(messages: List[str]) -> List[str]:
    all_messages = [message.replace("Atendente:", "").replace("Cliente:", "") for message in messages]
    
    # Filtra as mensagens que contêm os links do blipmediastore
    all_messages = [message for message in all_messages if "https://blipmediastore.blob.core.windows.net/secure-medias/" not in message]
    
    if not all_messages:
        return []
    
    # fazendo a vetorização do x e do lsa
    X = vectorizer.fit_transform(all_messages)
    svd = TruncatedSVD(n_components=1)
    normalizer = Normalizer(copy=False)
    lsa = make_pipeline(svd, normalizer)
    X_lsa = lsa.fit_transform(X)
    feature_names = vectorizer.get_feature_names_out()
    main_topic_words = [feature_names[i] for i in svd.components_[0].argsort()[:-10:-1]]
    
    # to removendo
    main_topic_keywords = [word for word in main_topic_words if word.lower() not in irrelevant_keywords and word.lower() not in ["senhor", "senhora", "intelbras", "aceito", "ok", "recomendo", "baseado", "olá"]]
    
    return main_topic_keywords


def get_top_words_by_speaker(messages: List[str]) -> dict:
    word_freq_by_speaker = {
        "Cliente": Counter(),
        "Atendente": Counter()
    }

    additional_stopwords = [
        "pra", "vc", "tb", "pq", "p", "ta", "to", "q", "a", "e", "o",
        "os", "as", "um", "uma", "uns", "umas", "ao", "aos", "na", "nas",
        "no", "nos", "em", "por", "para", "com", "se", "nao", "mais", "ja",
        "mas", "me", "te", "nos", "vos", "lhe", "lhes", "se", "que", "porque",
        "como", "onde", "quando", "isso", "isso", "essa", "esse", "isso", "isso",
        "estou", "estava", "estaremos", "esteja", "estivesse", "estiver", "estivessemos", "intelbras",
        "senhor", "senhora", "cliente", "atendente", "de", "nem", "ok"
    ]

    for message in messages:
        if ":" in message:
            speaker, content = message.split(":", 1)
            content = content.strip()
            doc = nlp(content)

            tokens = [token.text.lower() for token in doc if token.is_alpha]
            tokens = [token for token in tokens if token not in stop_words_lower]  # tira do CountVectorizer
            tokens = [token for token in tokens if token not in additional_stopwords]

            word_freq_by_speaker[speaker].update(set(tokens))  # Usar set() para contar palavras únicas

    top_words_by_speaker = {}
    for speaker, word_freq in word_freq_by_speaker.items():
        top_words = word_freq.most_common(3)
        top_words_with_counts = [{"word": word, "count": count} for word, count in top_words]
        top_words_by_speaker[speaker] = top_words_with_counts

    return top_words_by_speaker


def analyze_sentiment(messages: List[str]) -> str:
    positive_keywords = ["satisfeito", "excelente", "ótimo", "adorei", "incrível", "parabéns", "feliz", "recomendo", "maravilhoso", "fantástico", "surpreendente", "perfeito", "encantador", "impressionante", "ótima escolha", "top", "estou contente", "satisfação total", "muito bom", "espetacular"]
    negative_keywords = ["insatisfeito", "frustrado", "decepcionado", "problemas", "ruim", "péssimo", "dificuldades", "terrível", "desapontado", "horrível", "odiei", "não recomendo", "arrependido", "muito ruim", "desastroso", "lamentável", "descontente", "péssima escolha", "muito insatisfatório", "causou problemas"]

    positive_count = 0
    negative_count = 0

    # Processa cada mensagem da conversa
    for message in messages:
        blob = TextBlob(message.lower()) 
        sentiment = blob.sentiment.polarity

        for keyword in positive_keywords:
            if keyword in message.lower():
                positive_count += 1

        for keyword in negative_keywords:
            if keyword in message.lower():
                negative_count += 1
    
    if positive_count > negative_count:
        return "positivo"
    elif negative_count > positive_count:
        return "negativo"
    else:
        return "neutro"

@app.post("/analyze_topic/")
def analyze_topic(conversation: ConversationInput):
    main_topic_keywords = get_main_topic(conversation.messages)
    sentiment = analyze_sentiment(conversation.messages)
    
    if sentiment == "positivo":
        sentiment_response = "Sentimento positivo."
    elif sentiment == "negativo":
        sentiment_response = "Sentimento negativo."
    else:
        sentiment_response = "Sentimento neutro."
    
    main_topic_keywords = main_topic_keywords[:2]
    
    topic_response = f"Os tópicos principais da conversa são: '{main_topic_keywords[1]}' e '{main_topic_keywords[0]}'. {sentiment_response}"
    
    return {"response": topic_response}


@app.post("/top_words_by_speaker/")
def top_words_by_speaker_endpoint(conversation: ConversationInput):
    top_words = get_top_words_by_speaker(conversation.messages)
    return {"top_words_by_speaker": top_words}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3009)
