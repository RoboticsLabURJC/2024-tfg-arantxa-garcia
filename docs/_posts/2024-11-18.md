---
title: "Progreso 18 Nov"
last_modified_at: 2024-11-18T09:24:00
categories:
  - Blog
tags:
  - Progreso 18 Nov
---

# Cambio en el json

Hasta este momento había guardado el json con una estructura en la que separaba en distintos campos cads punto ( en 'x', 'y' e index, por ejemplo). Como vimos que esto podía dar problemas para futuro decidimos que lo mejor sería reformularlo como arrays, lo que haría el procesamiento futuro mucho más sencillo. Al cambiar esto, tuve que modificar prácticamente todas mis clases porque estaban pensadas para coger los datos de la manera anterior. 

# Balancer y su reconstructor

Queríamos ahora sacar un nuevo dataset con los datos balanceados. Es decir, que todas las clases contaran con la misma cantidad de frames. Además, había que tener en cuenta que los frames elegidos fueran válidos, es decir, que si cojo frames en los que el conductor está conduciendo solo con la mano izquierda, es necesario asegurarse de que tenemos detectada la mano izquierda en ese frame. 

En ese nuevo dataset necesitábamos también las imágenes de los frames a los que pertenece realmente, así que en el programa lo que hacíamos era crear un nuevo directorio en el que íbamos guardando las imágenes catalogándolas por frame al que peretenecían originalmente, sesión en la que se grabaron y qué cámara era(face, pose o hands). Así, a la hora de reconstruirlo era muy fácil acceder a ellas y mucho más rápido que si tuvieramos que buscar el frame en el video.

En el reconstructor vi que la cara tiende a desincronizarse, tengo que arreglar eso.

# Gaze

Hice un programa para ver si el gaze mejora si recortamos la cara pero no parece que ese sea el caso.
