# Is your conten toxic?

# **Introduction**

Данный сервис создан с целью определения токсичности того или иного высказывания (комментария). 

Его использование может быть полезно сайтам, на которых разерешено оставлять комментарии. Скажем, мы хотим посмотреть отзывы 
о некотором ресторане. На популярных сайтах активные пользователи оставляют сотни комментариев, однако среди них могут оказаться отзывы с нецензурной лексикой. Такие комментарии обычно не несут содержательный характер, также могут кого-то оскроблять, поэтому модераторы 
будут вынуждены удалить такой отзыв с сайта. При большом потоке пользователей разумно использовать сервисы, которые смогут проверять отзыв на токсичность. 


Данные для его создания были взяты с соревнования на kaggle.com "Jigsaw Unintended Bias in Toxicity Classification". <br>
Сервис способен принимать POST запрос со строкой в формате Json и возвращать 'Toxic' или 'Ok' также в формате Json. <br>

Для обработки слов (перевод в числовые значения) в строке я использовал статистику Tf-idf. Обученный на данных вокабуляр находится в файле _vectorize.pkl_. <br>
Далее я обучил логистическую регрессию (roc_auc_score ~0.9) и сохранил классификатор в файл _log_reg_model.pkl_. Код с обработкой данных и обучением модели находится в файле _log_reg_model.py_.

Ниже представлена интсрукция по запуску веб-сервиса. Для запуска сервиса на вашем компьютере вам нужен только Docker, все нужные файлы он установит самостоятельно.

# **Libraries**

Для создания данного веб-сервиса я использовал следующие Python библиотеки: <br>
_ML_: pandas, numpy, scikit-learn. _Web_: Flask

# **Step 1**

Клонируем себе репозиторий.
Открываем docker, заходим в папку python_web_service_with_model: <br>
`$ cd .../downloads/Is-toxic-web-service-master`
Запускаем команду в терминале: <br>
`$ docker-compose up --build`

Построили и открыли контейнер. Должен был образоваться локальный хост: http://localhost:5000/

Далее работаем с моим веб-сервисом.

# **Step 2**

В файле _service.py_ находится функция, которая способна принимать POST запрос с Json (комментарием) и возвращать 'Toxic' или 'Ok' 
также в формате Json.

Есть два способа отправить данный POST запрос:
1. Используя curl
 
Для этого в терминале нужно вбить следующее: <br>
`curl --header "Content-Type: application/json" \
  --request POST \
  --data '{'Comment': 'comment text'}' \
  http://localhost:5000/is_toxic`
  
Здесь _data_ - данные, которые мы хотим, чтобы наш сервис обработал.
На выходе мы получим токсичен комментарий или нет.

2. С помощью Postman

В этой программе нужно сделать POST запрос через адрес http://localhost:5000/is_toxic.
В `Headers` в `Key` добавить `Content-Type`, а в `Value` - `application/json`. Далее в `Body` нужно поставить галочку возле `raw`, 
в поле ниже вбить текст в формате Json. <br>
Пример того, как должны выглядеть параметры в Postman: [Postman][1]

[1]: https://yadi.sk/d/62KEBt5xRGF5EQ/postman_example.png "Postman"

