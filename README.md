# Named Entity Recognition

# Атауларды тану (NER) мини-жоба

Бұл жоба — алдын ала оқытылған трансформер моделін қолданып, сөйлемдегі атауларды (атаулы құрылымдарды) анықтайтын шағын NLP жобасы. Қосымша Streamlit және Hugging Face Transformers кітапханасына негізделген.

## 🔍 Не істейді?

- Пайдаланушы енгізген (ағылшын тіліндегі) сөйлемді қабылдайды
- Келесі категориялар бойынша атауларды анықтайды:
  - **PER** – адам есімі
  - **LOC** – орын, географиялық атаулар
  - **ORG** – ұйым атауы
  - **MISC** – басқа атаулар
- Табылған атаулар мен олардың түрін көрсетеді

## 🧠 Қолданылған модель

Ағылшын тіліндегі мәтіндермен жұмыс істейтін Hugging Face моделдері, мысалы:
- `dslim/bert-base-NER`

# Мини-проект распознавания именованных сущностей (NER)

Это небольшой проект обработки естественного языка (NLP), который выполняет распознавание именованных сущностей (NER) с использованием предварительно обученной модели Transformer. Приложение создано с использованием Streamlit и библиотеки Transformers от Hugging Face.

## 🔍 Что делает

- Принимает пользовательский ввод (предложение на английском языке)
- Применяет модель NER для обнаружения таких сущностей, как:
- **PER** (человек)
- **LOC** (местоположение)
- **ORG** (организация)
- **MISC** (разное)
- Отображает обнаруженные сущности и их метки

## 🧠 Используемая модель

Мы используем модель Transformer от Hugging Face, которая поддерживает английский язык для задач NER, таких как:
- `dslim/bert-base-NER` или аналогичная

# Named Entity Recognition (NER) Mini Project

This is a small NLP project that performs Named Entity Recognition (NER) using a pre-trained transformer model. The app is built with Streamlit and Hugging Face's Transformers library.

## 🔍 What It Does

- Takes user input (a sentence in English)
- Applies a NER model to detect entities such as:
  - **PER** (person)
  - **LOC** (location)
  - **ORG** (organization)
  - **MISC** (miscellaneous)
- Displays the detected entities and their labels

## 🧠 Model Used

We use a Hugging Face transformer model that supports English for NER tasks, such as:
- `dslim/bert-base-NER` or similar
