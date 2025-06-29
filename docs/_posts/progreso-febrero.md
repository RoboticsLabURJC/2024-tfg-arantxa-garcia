---
title: "Progreso febrero"
last_modified_at: 2025-02-28T11:57:00
categories:
  - Blog
tags:
  - Progreso febrero
---

Este mes empecé las prácticas así que avancé un poco menos con el TFG. Igualmente os cuento a continuación los avances que hice:

Ya con el test y train separado en distintas personas, buscamos mejorar al máximo posible el dataset para conseguir mejores resultados. Las primeras matrices de test y train nos dan los siguientes resultados:

![Primera matriz confusion test y train](image-3.png)

Esta matriz nos ayudó a decidir que sí podíamos conseguir un modelo que generalizase bien con todas las acciones, aunque todavía no funcionase bien del todo por la acciones de solo mano derecha.

Lo siguiente que decidimos fue añadir ruido gaussiano y una traslación para tener más datos y que pudiese generalizar mejor. Gracias a eso obtuvimos la siguiente matriz:

![Matriz de confusion con ruido](image-4.png)

Aún fallaba en la acción derecha pero al menos el train mejoraba