import spacy
from collections import Counter
from fastapi import FastAPI, HTTPException
from typing import List
from pydantic import BaseModel
from textblob import TextBlob

app = FastAPI()

nlp = spacy.load("pt_core_news_sm")

class ConversationInput(BaseModel):
    messages: List[str]

def get_main_topic(messages: List[str]) -> str:
    all_tokens = []

    for message in messages:
        doc = nlp(message)
        tokens = [token.text for token in doc if token.is_alpha]
        all_tokens.extend(tokens)

    # Calcula as frequências das palavras
    word_freq = Counter(all_tokens)

    # Encontra a palavra mais comum como tópico principal
    main_topic = word_freq.most_common(1)
    
    return main_topic[0][0] if main_topic else "Tópico não identificado"

def analyze_sentiment(messages: List[str]) -> str:
    positive_keywords = ["satisfeito", "excelente", "ótimo", "adorei", "incrível", "parabéns", "feliz", "recomendo"]
    negative_keywords = ["insatisfeito", "frustrado", "decepcionado", "problemas", "ruim", "péssimo", "dificuldades"]

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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=3009)
