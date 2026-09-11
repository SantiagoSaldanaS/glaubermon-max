> Prioridad actual (11 de septiembre): **alinear los entornos antes de reentrenar**. El entrenamiento está pausado por código. Ver [estado de alineación](alignment/ESTADO.md). Los pilotos descritos debajo son anteriores a esta pausa.

# Estado actual: correcciones, entrenamiento oficial y piloto externo

Actualizado: 11 de septiembre de 2026. Rama `codex/simulator-training-corrections`.

## Decisión implementada

La red, el evaluador híbrido y la búsqueda siguen en Python. No se hizo una reescritura. El motor oficial de Pokémon Showdown (Node, paquete npm 0.11.11) arbitra la nueva ruta de self-play y la evaluación. El motor interno mantiene las correcciones anteriores y queda como backend explícito `internal-privileged`, útil para experimentos; todavía no equivale a Showdown.

El entrenamiento predeterminado usa `showdown`, con observaciones construidas por el mismo `ShowdownBot` que juega en vivo. Cada jugador recibe únicamente su solicitud privada y su canal oficial filtrado; las hipótesis sobre movimientos, objetos y estadísticas del rival siguen siendo estimaciones del cliente. Nunca se entrega al decisor el objeto Battle del simulador ni la solicitud privada contraria. La búsqueda interna continúa siendo aproximada, pero las consecuencias que generan las etiquetas las decide Showdown.

Se conservan los pesos originales. Las salidas nuevas van a `runs/rebel-official-v2` o a una carpeta de experimento explícita. No se lanzó un entrenamiento masivo ni se sustituyó el checkpoint de juego.

## Correcciones de esta tanda

- Las partidas con límite de turnos tienen resultado ausente (`None`). Su política puede usarse para imitar la búsqueda, pero no aporta pérdida de valor. Un lote completamente truncado no actualiza la cabeza de valor ni por weight decay. Un empate oficial terminal sí vale cero.
- Se recogen ejemplos desde la perspectiva propia de ambos jugadores. Las decisiones normales usan la distribución Nash; los reemplazos forzosos imitan la selección del cliente. Los dos jugadores resuelven desde su propia información, por lo que no se usa la optimización de sacar ambas acciones de una única matriz omnisciente.
- Por defecto se permiten 300 turnos; profundidad de entrenamiento configurable, inicialmente 1. Las trazas oficiales por partida se conservan en `rollouts/`. Una acción rechazada o un timeout aborta esa recolección antes de incorporarla al buffer.
- Se separan dos equipos de práctica nuevos de los tres equipos congelados de evaluación. Todos pasan la validación Gen 9 OU. La cobertura del pequeño pool es limitada y no se considera suficiente para una campaña larga. El checkpoint histórico podría haber visto los equipos de evaluación; no se presume independencia de esos datos antiguos.
- El lector conserva el turno público, respeta solicitudes con menos de cuatro movimientos (incluido Struggle) y excluye cambios cuando la solicitud oficial indica `trapped`.
- Se incorporó el push `f72b0ea` de optimización, corrigiendo su cache para incluir todos los atributos codificados y limitarlo a 4096 entradas. Se restauró el indicador de KO en los espacios vacíos del tensor. No se cambió el default híbrido a heurístico ni se adoptó el truncado de experiencias del nuevo modo por pasos.
- Reanudar exige el mismo contrato de datos: backend, versión oficial, pool de equipos, profundidad y límite de turnos. Se restauran pesos y contadores; el optimizador y replay buffer se reinician, lo cual sigue siendo una limitación explícita.

## Ejecutar

Desde la raíz del repositorio:

```sh
python -m pip install -e '.[dev]'
npm --prefix tools/showdown ci
# El rival del benchmark usa una revisión fijada:
python -m pip install -r glaubermon/evaluation/requirements.txt
python -m pytest tests/ -q
python -m glaubermon.scripts.train_rebel --games 2 --checkpoint-dir runs/official-pilot-new --mechanics-seed 7331 --max-turns 150 --showdown-path tools/showdown/node_modules/pokemon-showdown
python -m glaubermon.evaluation.official_benchmark --showdown tools/showdown/node_modules/pokemon-showdown --output runs/evaluation-new --games 100 --depth 2 --jobs 4
python -m glaubermon.evaluation.summarize_benchmark runs/evaluation-new
```

El package-lock oficial se incluye en `tools/showdown/`. El workspace actual también conserva la instalación original en `../showdown-parity/`, aceptada por `--showdown-path`. No se necesitan cuentas para estas pruebas. El transporte local no hace login ni publica replays.

## Validación y alcance

- Pasan 134 tests. Se prueban aislamiento de canales oficiales, reproducibilidad de semillas, rechazo de equipos ilegales, invariancia de observaciones ante sets secretos diferentes, revelación de movimientos, etiquetas de truncado y actualización real del optimizador.
- El piloto de entrenamiento con los pesos originales completó dos partidas oficiales: 48 turnos y 117 ejemplos entre ambos lados, con una actualización real. Tardó aproximadamente 12–15 s en repeticiones en CPU, profundidad 1, con otros procesos de evaluación activos. El checkpoint final del piloto está en `runs/official-training-ready-v2/`; se verificó que todos sus parámetros son finitos y que reanudar recupera exactamente pesos y contadores. Esto verifica ejecución, no superioridad del nuevo checkpoint.
- Microbenchmark de codificación: mediana de 200 codificaciones del mismo estado 6v6 de 132.6 ms a 92.7 ms (aprox. 30% menos tiempo), siete repeticiones intercaladas con cache caliente. Los tensores coinciden exactamente. No representa una aceleración de todo el entrenamiento.
- La evaluación externa congela `01a6ee1` y el checkpoint original: 100 partidas híbridas y 100 heurísticas, profundidad 2, intercambiando equipos/lados por bloque. Resultado: híbrido 73/100 y heurísticas 79/100; diferencia −6 puntos (IC 95% −16 a +4), sin acciones inválidas. Ver [informe del piloto](benchmarks/pilot-01a6ee1/RESULTADOS.md). Las trazas completas quedan en `../evaluation-results/official-pilot-01a6ee1/`. Esa versión es anterior a las últimas correcciones del lector y al entrenamiento de prueba.

No se certifica equivalencia del motor rápido con Showdown. Siguen pendientes sus mecánicas incompletas, señales del cliente aún ausentes (por ejemplo pantallas y Trick Room), mejores creencias sobre el rival, ampliar los equipos de práctica, reanudar el estado completo de optimización y evaluar el checkpoint nuevo. Metamon y Foul Play quedan para la comparación siguiente; este piloto contra SimpleHeuristicsPlayer no certifica SOTA.

---

# Historial: primera tanda (ba47370)

Rama: `codex/simulator-training-corrections`, basada en `69ff547`.

## Cambios

- Las ramas de búsqueda ya no comparten objetos Move ni sus diccionarios de boosts. Clonar conserva flags de ejecución y no vuelve a ejecutar la inicialización de Pokémon ni a reinterpretar formas desde el Dex.
- Las transiciones consumen PP cuando se intenta el movimiento, también al fallar; no consumen PP si el usuario cae antes de actuar o permanece dormido. Se contempla Pressure y Struggle al agotarse las opciones, con daño sin tipo y recoil.
- `simulate_turn_transition(..., sample_outcomes=True, rng=random.Random(seed))` sortea empates de velocidad, precisión básica, críticos, roll de daño y Protect. Un fallo de Protect reinicia la racha. Los estados de protección se limpian al siguiente turno.
- El orden básico del daño usa el roll entero antes de STAB y efectividad. Los críticos ignoran reducciones ofensivas e incrementos defensivos correspondientes. Los 16 daños normales y críticos de Tackle del fixture Snorlax se contrastaron con el motor oficial.
- La búsqueda conserva por defecto sus aproximaciones de daño/Protect; generar partidas ahora usa el modo de resultados sorteados. Esto separa una estimación de búsqueda de una transición de rollout.
- ReBeL genera sus decisiones con HybridEvaluator y vuelve a `model.eval()` después de cada actualización para que dropout no afecte esa búsqueda. El benchmark interno también usa el híbrido.
- Las transiciones sorteadas dejan el activo derrotado en su sitio. El entrenador ejecuta una fase separada de reemplazo, incluyendo cadenas de KOs por hazards. Invertir perspectivas conserva también los contadores del campo y Trick Room.
- La salida por defecto de ReBeL es `runs/rebel-corrections`, separada de los checkpoints entregados. Una carpeta nueva puede partir de los pesos ReBeL originales sin heredar su contador de partidas. Los metadatos identifican este modo como experimental. Un checkpoint incompatible o contadores de otro modo detienen la ejecución, evitando continuar silenciosamente con pesos aleatorios o mezclar experimentos.
- Se declaran aiohttp y tqdm, requeridos por el cliente/entrenador, y se limita el descubrimiento de paquetes a glaubermon.

## Validación realizada

- 123 tests pasan en 3.78 s, incluidos los 106 originales, aislamiento de ramas, PP, distribuciones con semillas, Protect, Struggle, reemplazos y rechazo de pesos incompatibles.
- Una partida 6v6 usando los pesos entregados terminó en 22 turnos. El rollout tomó aproximadamente 3.81 s con PyTorch CPU a un hilo en este Mac. Un paso real del optimizador produjo pérdidas finitas; los pesos temporales se descartaron. Esto comprueba ejecución, no mejora de juego.
- El banco externo `../showdown-parity/` conserva los 320 casos oficiales y las comparaciones previas. `comparison-sampled-v1.json` registra el nuevo modo y el hash del diff para identificar modificaciones aún no comprometidas.
- Medición inicial de 1,000 clones: mediana aproximada de 33.4 ms antes y 18.2 ms después. En cinco búsquedas depth 1, la mediana pasó aproximadamente de 155.8 a 169.7 ms. La copia mejora, pero no se afirma una aceleración total: se agregó trabajo para corregir mecánicas.

## Comandos históricos de la primera tanda

Desde la raíz del repositorio, en un entorno con sus dependencias:

```sh
python -m pytest tests/ -q
python -m glaubermon.scripts.train_rebel --games 1 --checkpoint-dir runs/pilot-001 --mechanics-seed 7
```

Una ejecución de una sola partida puede no llenar el mínimo de 64 muestras requerido por el bucle para actualizar pesos; la prueba automática ejercita explícitamente una actualización. `--mechanics-seed` controla los sorteos de las transiciones, no las semillas globales de equipos, políticas, NumPy o PyTorch. Reanudar sigue restaurando pesos y contadores, no el optimizador ni el replay buffer completos.

## Límites identificados en la primera tanda

Este motor sigue siendo parcial: no se certifica equivalencia completa con Showdown ni se recomienda una corrida masiva todavía. Falta revisar, entre otros, el turno de despertar y Sleep Talk, parálisis/congelación, efectos secundarios, reglas de habilidades/objetos que alteran precisión o críticos, pivotes/phazing, eliminación de boosts al salir y varios modificadores/redondeos de daño. La política puede aprender a explotar cualquier regla pendiente.

ReBeL aún usa un límite de 35 turnos y una etiqueta heurística cuando no hay ganador. Las observaciones de self-play incluyen información del rival que no necesariamente está disponible en vivo. Esos objetivos/observaciones deben corregirse o contrastarse con el entorno oficial antes de un entrenamiento largo. Los demás scripts de entrenamiento heredados no se migraron en esta tanda.

El siguiente paso es la evaluación externa del híbrido contra SimpleHeuristicsPlayer en Showdown, para tener una referencia de calidad independiente de este simulador. Las correcciones restantes y la optimización de inferencia se priorizarán con esa evidencia; Rust/C++ sigue pospuesto.
