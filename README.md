# Цифровой прорыв 2024 СЗФО
# Кейс от НИИАС РЖД: Интеллектуальный пульт
<a name="readme-top"></a>
<p align="center">  
<img width="30%" src="./images/photo_2024-10-06_07-08-13.jpg" alt="banner">
</p>
  <p align="center">
    <!--<h1 align="center">LLAIM</h1>-->
  </p>
  <p align="center">
    <p></p>
    <!-- <p><strong>Интеллектуальный пульт составителя.</strong></p> -->

  </p>
</div>

**Содержание:**
- [Проблематика задачи](#title1)
- [Описание решения](#title2)
- [Тестирование решения](#title3)

### Разработан в рамках [ЦП](https://hacks-ai.ru/events/1077380) командой "LLAIM".

### Live версия доступна [тут](https://t.me/rosatom_support_bot)

## <h3 align="start"><a id="title1">Проблематика задачи</a></h3>  
1. Необходимость голосового управления
   
   1.1 шумная окружающая среда
   
   1.2 Высокий риск неверного распознавания команды
   
3. Упрощение управления объектом
   
   2.1 трансляция речечых команд
   
4. Ограниченность вычислительных ресурсов
   
    3.1 решение должно работать на микроконтроллере
   
    3.2 важная скорость end-to-end обработки 
----

### Поиск по структурированным данным (схожие вопросы на основе истории обращений)


Поиск информации по заданной тематике на основе структурированных данных (таблицы).

----

## <h3 align="start"><a id="title2">Описание решения</a></h3>

<img src="./resources/photo_2024-06-16_08-42-59.jpg" alt="Архитектура решения" width="700"/>

* Шумоподавление 
* ASR Wave2vec
* Поиск команды из заданного набора в тексте
* Вывод в формате .json
 
### Структура проекта

```
├── app # основная директория проекта
│   ├── utils # содержит утилиты для работы проекта
│   ├── main_app.py # тг бот
│   ├── chat_bot.py # файл ответов на вопросы в формате excel
│   ├── app.ipynb # demo бота с gradio
├── data # содержит данные и БД для проекта, а также тестовые вопросы
├── README.md
├── requirements.txt
└── resources # ресурсы проекта
```

## <h3 align="start"><a id="title3">Тестирование решения</a></h3> 

## Development

0. Install requirements

```
pip install -r requirements.txt
```
