# Imagem base Python
FROM python:3.10-slim

# Define o diretório de trabalho no contêiner
WORKDIR /app

# Copie os arquivos necessários para o contêiner
COPY . .

# Instale as dependências especificadas no requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Exponha a porta em que o aplicativo Streamlit estará em execução
EXPOSE 3009

# Execute o comando run para iniciar o aplicativo
CMD ["python", "main.py"]
