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
| Unit 3 | [Потоковые генеративные модели](https://github.com/sswt/dive2gai/tree/main/unit3) | <ul><li>[Flow-based generative models](https://github.com/sswt/dive2gai/tree/main/unit3/NormalizingFlowsIntro.ipynb)</li><li>[Dive to Normalizing Flows🤓](https://github.com/sswt/dive2gai/tree/main/unit3/DiveToNF.ipynb)</li><li>Continious Normalizing Flow</li></ul> | 2023-09-28 |
| Unit 4 | [Генерация последовательностей](https://github.com/sswt/dive2gai/tree/main/unit1) | <ul><li>RNN, LSTM</li><li>Professor forcing🤓</li><li>SeqGAN</li><li>LeakGAN</li></ul> | 2023-10-05 |
| Unit 5 | Трансформеры, механизм внимания | <ul><li>Attention is all you need</li><li>Self-attention, transformers</li><li>T5, GPT</li></ul> | 2023-10-19 |
| Unit 6 | Диффузные модели | <ul><li>Denoising Diffusion Models</li><li>Fine-Tuning and Guidance</li><li>Stable Diffusion</li></ul> | 2023-11-02 |
| Unit 7 | Метрики качества генерации |  |  |
| Unit 8 | Reinforcement learning для генеративных моделей |  |  |
| Unit 9 | Energy-based models |  |  |
| Unit 10 | Мультимодальные генеративные модели |  |  |
| Unit 11 | Self-supervised learning |  |  |

Готовые короткие курсы по теме:

* [Generative AI learning path](https://www.cloudskillsboost.google/journeys/118) от Google
* [Generative AI with Large Language Models](https://www.coursera.org/learn/generative-ai-with-llms) на Coursera

Книги:

* Tomczak, Jakub. Deep Generative Modeling. Springer, 2022
* Foster, David. Generative Deep Learning, 2nd ed., O'Reilly, 2023
