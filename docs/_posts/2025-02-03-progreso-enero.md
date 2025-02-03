---
title: "Progreso enero"
last_modified_at: 2025-02-03T11:57:00
categories:
  - Blog
tags:
  - Progreso enero
---

# Balancer

Para el balancer ahora hemos hecho unas cuantas mejoras para que cogiera datos más válidos. Primero que todo, como el dataset está etiquetado al milímetro, teníamos casos en los que un frame estaba etiquetado como que solo estaba la mano izquierda en el volante pero aún tenía ambas, lo que podía resultar en que en el entrenamiento hubiera frames identicos con etiquetas diferentes. Para evitar esto lo que decidimos fue quitar los 10 primeros y últimos frames de cada accion del dataset.

# Multilabel vs multiclass

Aunque teníamos un modelo multilabel con 6 clases distintas no nos estaba dando muy buenos resultados así que para ver por qué decidimos dar un pasito atrás y volver a un modelo multiclass solo de 3 clases (las de las manos en el volante). Esto ha resultado en que obviamente tengamos resultados muchísimo mejores, pues son 3 clases excluyentes entre sí. Aun así, nos sirvió para darnos cuenta de que un dataset variado y completo es súper importante para los resultados, por lo que mi papel ahora mismo es tratando de mejorar el dataset que usamos.

# Cambio de persona

En el dataset estabamos usando las mismas sesiones para train y para test por lo que no podemos saber si nuestro modelo realmente generaliza. Esto se debe a que es probable que en train y test existiera el mismo frame repetido, por lo que el modelo no se enfrenta a nada nuevo para predecir.

Para evitar esto, hemos separado unas sesiones para los datos de train y otra sesion de una persona nueva para test.

# Scripts python para jsons

He creado unos scripts en python para ayudarme a tratar los json. De momento tengo el concatjsons para unir dos jsons de dos sesiones distintas. También tengo el reducer, que, por ejemplo, si le doy como argumento 3, coge uno de cada tres frames. Tengo pendiente hacer uno que coja directamente los n primeros.

# Recordatorio de uso

Viendo que ya había hecho bastantes programas y que no me acordaba de como se usaban todos, me he dedicado a poner al principio del archivo su uso.
