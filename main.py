import spacy
from collections import Counter, defaultdict
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from textblob import TextBlob
from spacy.lang.pt.stop_words import STOP_WORDS

app = FastAPI()

nlp = spacy.load("pt_core_news_sm")

class ConversationInput(BaseModel):
    messages: List[str]

def get_main_topic(messages: List[str]) -> str:
    all_tokens = []

    for message in messages:
        # Remova os prefixos "atendente:" e "cliente:" e, em seguida, tokenize a mensagem
        message = message.replace("Atendente:", "").replace("Cliente:", "")
        doc = nlp(message)
        
        # Filtra palavras que são alfanuméricas e não estão na lista de stopwords
        tokens = [token.text for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
        all_tokens.extend(tokens)

    # Calcula as frequências das palavras
    word_freq = Counter(all_tokens)

    # Encontre a palavra mais comum como tópico principal
    main_topic = word_freq.most_common(1)
    
    return main_topic[0][0] if main_topic else "Tópico não identificado"

def get_top_words_by_speaker(messages: List[str]) -> dict:
    word_freq_by_speaker = defaultdict(Counter)

    for message in messages:
        if ":" in message:
            speaker, content = message.split(":", 1)  # Separa o falante do conteúdo
            content = content.strip()  # Remove espaços extras
            doc = nlp(content)
            
            # Filtra palavras que são alfanuméricas e não estão na lista de stopwords
            tokens = [token.text.lower() for token in doc if token.is_alpha and token.text.lower() not in STOP_WORDS]
            
            word_freq_by_speaker[speaker].update(tokens)

    top_words_by_speaker = {}
    for speaker, word_freq in word_freq_by_speaker.items():
        top_words = word_freq.most_common(3)
        top_words_by_speaker[speaker] = [word for word, _ in top_words]
    
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
    main_topic = get_main_topic(conversation.messages)
    sentiment = analyze_sentiment(conversation.messages)
    
    if sentiment == "positivo":
        sentiment_response = "Sentimento positivo."
    elif sentiment == "negativo":
        sentiment_response = "Sentimento negativo."
    else:
        sentiment_response = "Sentimento neutro."

    response = f"O tópico principal da conversa é '{main_topic}'. {sentiment_response}"
    return {"response": response}

@app.post("/top_words_by_speaker/")
def top_words_by_speaker_endpoint(conversation: ConversationInput):
    top_words = get_top_words_by_speaker(conversation.messages)
    return {"top_words_by_speaker": top_words}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3009)
