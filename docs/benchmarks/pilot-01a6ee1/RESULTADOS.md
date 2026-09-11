# Piloto oficial Gen 9 OU

Versión evaluada: `01a6ee10fd9381114472c7ad9871feb2748847a4`; Showdown `0.11.11`; poke-env `0.16.1`.
Profundidad 2; 4 procesos, PyTorch a 1 hilo por proceso. Checkpoint original respaldado; ningún peso entrenado con estas partidas.

| Modo | Victorias | Derrotas | Empates | Sin resolver | Puntuación | IC 95% por bloques |
|---|---:|---:|---:|---:|---:|---|
| hybrid | 73 | 27 | 0 | 0 | 73.0% | 65.0%–81.0% |
| heuristic | 79 | 21 | 0 | 0 | 79.0% | 70.0%–87.0% |

Diferencia emparejada híbrido − heurísticas: -6.0 puntos porcentuales; IC 95% -16.0 a 4.0.

## Integridad y latencia

- hybrid: 4061 decisiones, 0 registros de acciones inválidas, 0 elecciones default, 185491 llamadas reales a la red. Latencia p50 1.025 s; p95 1.599 s; máximo 3.918 s.
- heuristic: 3957 decisiones, 0 registros de acciones inválidas, 0 elecciones default, 0 llamadas reales a la red. Latencia p50 0.184 s; p95 0.361 s; máximo 0.829 s.

## Alcance

Son 50 bloques de dos partidas por modo. Dentro del bloque se intercambian agentes entre lados y equipos, conservando semilla y posiciones iniciales. El bootstrap remuestrea 50 bloques completos, con 20,000 réplicas; la diferencia entre modos usa los mismos bloques.

Solo se usaron los tres equipos entregados que pasan la validación OU de esta versión: balance, stall y pelol. Hyper offense fue excluido antes de congelar el piloto porque Roaring Moon está prohibido. Los equipos, semillas, configuración, hashes y política de fallos están en manifest.json. Cada partida conserva resultado, latencias, elecciones, canales privados separados y log público.

La incertidumbre describe este conjunto reducido de equipos y semillas. No mide toda la ladder ni certifica SOTA. El historial de entrenamiento del checkpoint original puede incluir estos equipos; no se afirma que fueran desconocidos para ese modelo. Estas partidas quedan reservadas para evaluación y no alimentan el nuevo entrenador.

Un resultado sin terminar conserva estado sin resolver; nunca se convierte en empate. Los fallos de infraestructura no se ocultan. La versión evaluada se congeló antes de las correcciones posteriores del lector y de la optimización de tensores; los pesos del entrenamiento de prueba no se evalúan aquí.

## Interpretación para el siguiente entrenamiento

El punto estimado favorece al modo heurístico por 6 puntos, pero el intervalo emparejado incluye cero. Este piloto no establece que el checkpoint neuronal actual mejore las heurísticas. Tampoco demuestra que toda red o arquitectura híbrida sea inferior. Se conserva el modo híbrido del proyecto y los pesos entregados; no se promueve el checkpoint del entrenamiento de prueba.

El siguiente experimento debe comparar un modelo reentrenado con observaciones permitidas y resultados terminales oficiales, usando equipos de práctica distintos y una evaluación reservada. El autojuego oficial corto ya pasó una actualización real y reanudación de pesos/contadores. Hace falta ampliar los equipos y medir al nuevo candidato antes de invertir en una corrida larga o en mayor profundidad.

Los resultados individuales están en `games.json`; los hashes de todas las trazas están en `raw_sha256.json`. Las trazas y los logs públicos de las 200 partidas permanecen en `evaluation-results/official-pilot-01a6ee1/` del workspace. No hubo publicación en Showdown ni partidas de ladder.
