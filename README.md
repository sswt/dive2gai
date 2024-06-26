# Dive into Generative AI

![Banner](https://github.com/sswt/dive2gai/assets/13690948/2787920a-db77-4d95-8e7b-1e8f2938ea86)

Самообразовательная инициатива по погружению в генеративный ИИ.

**Цель:** Познакомиться поближе с методами generative AI, местами погружаясь в детали их работы. Начиная от исторически самых первых генеративных моделей до востребованных на сегодняшний день.

**Два уровня погружения:**

* поверхностное, для общего понимания устройства моделей
* более глубокое в отдельные методы и алгоритмы (🤓)

Это просто компиляция материалов из различных открытых источников, локализованная и немного преобразованная под цели данного проекта. Программа имеет свой приблизительный таймлайн, идея состоит в том, чтобы раз в неделю выпускать что-то новое, приблизительно в соответствии с планом. Планируется, что инициативная группа авторов будет разбираться с темами в рамках программы и публиковать jupyter-ноутбуки как результат по каждой теме. Датасеты используются небольшие (MNIST, Fashion MNIST etc), чтобы была возможность запускать ноутбуки в Google Colab, например.

**Программа:**

| Номер  | Название       | Содержание     | 📆 Дата релиза  |
|--------|----------------|----------------|-----------------|
| Unit 0 | [Основы pytorch](https://github.com/sswt/dive2gai/tree/main/unit0) | Очень коротко про самые основы, для тех, кто впервые сталкивается с нейросетями | 2023-08-24 |
| Unit 1 | [Автоэнкодеры](https://github.com/sswt/dive2gai/tree/main/unit1)   | <ul><li>AE, VAE from scratch</li><li>ELBO 🤓</li></ul> | 2023-08-31 |
| Unit 2 | [Генеративные состязательные сети](https://github.com/sswt/dive2gai/tree/main/unit2)   | <ul><li>GAN, DCGAN, DCCGAN from scratch</li><li>Dive into GAN 🤓</li><li>WGAN, CycleGAN</li></ul> | 2023-09-14 |
| Unit 3 | [Потоковые генеративные модели](https://github.com/sswt/dive2gai/tree/main/unit3) | <ul><li>[Flow-based generative models](https://github.com/sswt/dive2gai/tree/main/unit3/NormalizingFlowsIntro.ipynb)</li><li>[Dive to Normalizing Flows🤓🤓](https://github.com/sswt/dive2gai/tree/main/unit3/DiveToNF.ipynb)</li></ul> | 2023-09-28 |
| Unit 4 | [Генерация последовательностей](https://github.com/sswt/dive2gai/tree/main/unit4) | <ul><li>RNN, LSTM</li><li>Professor forcing🤓</li><li>SeqGAN</li><li>LeakGAN</li></ul> | 2023-10-05 |
| Unit 5 | [Трансформеры, механизм внимания](https://github.com/sswt/dive2gai/tree/main/unit5) | <ul><li>Attention, transformers, GPT</li><li>Llama, quantization, fine-tuning</li></ul> | 2023-10-26 |
| Unit 6 | [Диффузные модели](https://github.com/sswt/dive2gai/tree/main/unit6) | <ul><li>Denoising Diffusion Models</li><li>Latent Diffusion Models</li></ul> | 2023-11-16 |
| Unit 7 | [Метрики качества генерации](https://github.com/sswt/dive2gai/tree/main/unit7) | <ul><li>CV</li><li>NLP</li></ul> | 2023-12-07 |
| Unit 8 | [Energy-based models](https://github.com/sswt/dive2gai/tree/main/unit8) | <ul><li>Deep Energy Based Models</li><li>Your models are secretly EBMs</li></ul> | 2023-12-14 |
| Unit 9 | [Генерация табличных данных](https://github.com/sswt/dive2gai/tree/main/unit9) | <ul><li>TVAE, CTGAN</li><li>TabDDPM, TabuLa</li></ul> | 2024-01-11 |
| Unit 10 | Мультимодальные генеративные модели | [Обзор эволюции мультимодальных моделей](https://github.com/sswt/dive2gai/tree/main/unit10)  | 2024-02-15 |
| Unit 11 | Закат трансформеров |  | 2024-03-28 |
| Unit 12 | [Непрерывные нормализующие потоки](https://github.com/sswt/dive2gai/tree/main/unit12) | <ul><li>Continious Normalizing Flows</li><li>Neural ODE & SDE models</li></ul> | 2024-03-14 |
| Unit 13 | Генерация табличных временных рядов | | 2023-04-04 |
| Unit 14 | Self-supervised learning |  | TBA |

Готовые короткие курсы по теме:

* [Generative AI learning path](https://www.cloudskillsboost.google/journeys/118) от Google
* [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) на Coursera

Книги:

* Tomczak, Jakub. Deep Generative Modeling. Springer, 2022
* Foster, David. Generative Deep Learning, 2nd ed., O'Reilly, 2023
