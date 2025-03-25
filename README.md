# Автокодировка открытых вопросов

### Для использования нужно включить VPN, т.к. для России возможности OpenAI не работают

### Как использовать:

* Сохранить нужный вопрос в формате csv в один столбец "Ответы". Файл назвать *responses.csv*.
* Если есть кодовый ключ, сохранить его в формате csv под двумя столбцами "Код" и "Категория". Файл назвать *codes.csv*.
* В файле *sys_prompts.json* следующие поля на актуальные: 
	subject: <тема исследования>, 
    first_question: <Вопрос, от которого могут зависеть ответы на нужный открытый. Например, оценка препарата для открытого "Почему именно так оценили">, 
    second_question: <Текст открытого вопроса>, 
    additional_requirements: <дополнительные комментарии, если качество кодировки не устраивает> 

* запустить скрипт *mar_response_classifier.py*.
* результат будет сохранен в файл *result.xlsx*.