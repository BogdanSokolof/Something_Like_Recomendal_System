# Базовый образ с conda
FROM continuumio/miniconda3:latest

# Устанавливаем рабочую директорию
WORKDIR /app

# Копируем environment.yml
COPY environment.yml .

# Создаём окружение из environment.yml
RUN conda env create -f environment.yml

# Делаем conda activate доступным в каждом RUN
SHELL ["conda", "run", "-n", "recsys", "/bin/bash", "-c"]

# Копируем остальные файлы проекта
COPY . .

# Открываем порт для Streamlit
EXPOSE 8501

# Запускаем Streamlit
CMD ["conda", "run", "--no-capture-output", "-n", "recsys", "streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]