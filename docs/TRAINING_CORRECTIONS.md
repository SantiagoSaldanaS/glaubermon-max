# Primera tanda de correcciones antes de reentrenar

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

## Cómo probar un piloto

Desde la raíz del repositorio, en un entorno con sus dependencias:

```sh
python -m pytest tests/ -q
python -m glaubermon.scripts.train_rebel --games 1 --checkpoint-dir runs/pilot-001 --mechanics-seed 7
```

Una ejecución de una sola partida puede no llenar el mínimo de 64 muestras requerido por el bucle para actualizar pesos; la prueba automática ejercita explícitamente una actualización. `--mechanics-seed` controla los sorteos de las transiciones, no las semillas globales de equipos, políticas, NumPy o PyTorch. Reanudar sigue restaurando pesos y contadores, no el optimizador ni el replay buffer completos.

## Límites y siguiente tanda

Este motor sigue siendo parcial: no se certifica equivalencia completa con Showdown ni se recomienda una corrida masiva todavía. Falta revisar, entre otros, el turno de despertar y Sleep Talk, parálisis/congelación, efectos secundarios, reglas de habilidades/objetos que alteran precisión o críticos, pivotes/phazing, eliminación de boosts al salir y varios modificadores/redondeos de daño. La política puede aprender a explotar cualquier regla pendiente.

ReBeL aún usa un límite de 35 turnos y una etiqueta heurística cuando no hay ganador. Las observaciones de self-play incluyen información del rival que no necesariamente está disponible en vivo. Esos objetivos/observaciones deben corregirse o contrastarse con el entorno oficial antes de un entrenamiento largo. Los demás scripts de entrenamiento heredados no se migraron en esta tanda.

El siguiente paso es la evaluación externa del híbrido contra SimpleHeuristicsPlayer en Showdown, para tener una referencia de calidad independiente de este simulador. Las correcciones restantes y la optimización de inferencia se priorizarán con esa evidencia; Rust/C++ sigue pospuesto.
