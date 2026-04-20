#Импортируем необходимые библиотеки
import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import numpy as np
import pandas as pd
import lightfm as lf
import nmslib
import pickle
import scipy.sparse as sparse
import plotly.express as px
import os 

# БАЗОВАЯ ПАПКА ПРОЕКТА (укажите ваш полный путь)
BASE_PATH = 'C:/Users/Lenovo/IDE/SFDS/data/RecSysN2'
RATINGS_PATH = 'C:/Users/Lenovo/IDE/SFDS/data/RecSysN2/data/ratings.csv'
BOOKS_PATH = 'C:/Users/Lenovo/IDE/SFDS/data/RecSysN2/data/books.csv'
EMBEDDINGS_PATH = 'C:/Users/Lenovo/IDE/SFDS/data/RecSysN2/item_embeddings.pkl'

@st.cache_data  # Используйте cache_data вместо cache
def reading_files(folder_name=f'{BASE_PATH}/data'):
    """
    Функция для чтения файлов.
    Возвращает два DataFrame с рейтингами и характеристиками книг.
    """
    folder_path = os.path.normpath(folder_name)
    ratings_path = os.path.join(RATINGS_PATH)
    books_path = os.path.join(BOOKS_PATH)
    
    # Проверка существования файлов
    if not os.path.exists(ratings_path):
        raise FileNotFoundError(f"Файл не найден: {ratings_path}")
    if not os.path.exists(books_path):
        raise FileNotFoundError(f"Файл не найден: {books_path}")
    
    ratings = pd.read_csv(ratings_path)
    books = pd.read_csv(books_path)
    
    st.success(f"Загрузено: {len(ratings)} оценок, {len(books)} книг")
    return ratings, books

def make_mappers(books):
    """
    Функция для создания отображения id в title и authors.
    Возвращает два словаря:
    * Ключи первого словаря – идентификаторы книг, значения – их названия.
    * Ключи второго словаря – идентификаторы книг, значения – их авторы.
    """
    name_mapper = dict(zip(books.book_id, books.title))
    author_mapper = dict(zip(books.book_id, books.authors))
    return name_mapper, author_mapper

def load_embeddings(file_name=f'{BASE_PATH}/item_embeddings.pkl'):
    """
    Функция для загрузки векторных представлений.
    Возвращает прочитанные эмбеддинги книг и индекс (граф) для поиска похожих книг.
    """
    if file_name is None:
        file_name = EMBEDDINGS_PATH
    
    if not os.path.exists(file_name):
        raise FileNotFoundError(f"Файл эмбеддингов не найден: {file_name}")
    
    with open(file_name, 'rb') as f:
        item_embeddings = pickle.load(f)
    
    st.info(f"Загружены эмбеддинги: {item_embeddings.shape}")
    
    # Тут мы используем nmslib, чтобы создать быстрый knn
    nms_idx = nmslib.init(method='hnsw', space='cosinesimil')
    nms_idx.addDataPointBatch(item_embeddings)
    nms_idx.createIndex(print_progress=True)
    return item_embeddings, nms_idx

def nearest_books_nms(book_id, item_embeddings, index, n=10):
    """
    Функция для поиска ближайших соседей.
    Параметры:
    - book_id: ID книги
    - item_embeddings: матрица эмбеддингов (ОБЯЗАТЕЛЬНЫЙ параметр!)
    - index: построенный индекс nmslib
    - n: количество соседей
    Возвращает n наиболее похожих книг и расстояние до них.
    """
    # Проверка, что book_id существует
    if book_id >= len(item_embeddings):
        raise ValueError(f"book_id {book_id} вне диапазона (0-{len(item_embeddings)-1})")
    
    nn = index.knnQuery(item_embeddings[book_id], k=n)
    return nn

def get_recomendation_df(ids, distances, name_mapper, author_mapper):
    """
    Функция для составления таблицы из рекомендованных книг.
    Возвращает DataFrame со столбцами:
    * book_name — название книги;
    * book_author — автор книги;
    * distance — значение метрики расстояния до книги.
    """
    names = []
    authors = []
    
    # Для каждого индекса книги находим её название и автора
    for idx in ids:
        if idx in name_mapper:  # Проверка существования
            names.append(name_mapper[idx])
            authors.append(author_mapper[idx])
        else:
            names.append(f"Unknown book (id: {idx})")
            authors.append("Unknown author")
    
    # Составляем DataFrame
    recomendation_df = pd.DataFrame({
        'book_name': names, 
        'book_author': authors, 
        'distance': distances
    })
    return recomendation_df

# ========== ЗАГРУЗКА ДАННЫХ ==========
ratings, books = reading_files(folder_name=f'{BASE_PATH}/data')
name_mapper, author_mapper = make_mappers(books)
item_embeddings, nms_idx = load_embeddings(file_name=f'{BASE_PATH}/item_embeddings.pkl')

# Создаем словарь для поиска book_id по названию книги
title_to_id = dict(zip(books['title'], books['book_id']))

# ========== ИНТЕРФЕЙС ==========
st.title("Recommendation System Of Books")

st.markdown("""Welcome to the web page of the book recommendation app!
This application is a prototype of a recommendation system based on a machine learning model.

To use the application, you need:
1. Enter the approximate name of the book you like
2. Select its exact name in the pop-up list of books
3. Specify the number of books you need to recommend

After that, the application will give you a list of books most similar to the book you specified""")

# Вводим строку для поиска книг
title = st.text_input('Please enter book name', '')
title = title.strip().lower()

# Выполняем поиск по книгам — ищем неполные совпадения
if title:
    output = books[books['title'].str.lower().str.contains(title, na=False)]
    
    if not output.empty:
        option = st.selectbox("Select the book you need", output['title'].values)
        st.markdown(f'You selected: "{option}"')
    else:
        st.warning("No books found matching your query. Please try a different title.")
else:
    st.info("Please enter a book title to search.")

# Проверяем, что поле не пустое
if 'option' in locals() and option:
    # Указываем количество рекомендаций
    count_recomendation = st.number_input(
        label="Specify the number of recommendations you need",
        value=10
    )
    
    # Находим book_id для выбранной книги
    val_index = title_to_id[option]
    
    # Находим count_recomendation+1 наиболее похожих книг
    ids, distances = nearest_books_nms(val_index, item_embeddings, nms_idx, count_recomendation+1)
    # Убираем из результатов книгу, по которой производился поиск
    ids, distances = ids[1:], distances[1:]
    
    # Выводим рекомендации к книге
    st.markdown('Most simmilar books are: ')
    # Составляем DataFrame из рекомендаций
    df = get_recomendation_df(ids, distances, name_mapper, author_mapper)
    # Выводим DataFrame в интерфейсе
    st.dataframe(df[['book_name', 'book_author']])
    
    # Строим столбчатую диаграмму
    fig = px.bar(
        data_frame=df,
        x='book_name',
        y='distance',
        hover_data=['book_author'],
        title='Cosine distance to the nearest books'
    )
    # Отображаем график в интерфейсе
    st.write(fig)