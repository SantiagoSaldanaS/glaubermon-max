# Alineación con Showdown: abierta; reentrenamiento pausado

Instrucción de Felipe/Santiago, 11 de septiembre de 2026: corregir las diferencias de entorno antes de reentrenar. `train_rebel.train()` ahora rechaza una corrida mientras `docs/ALIGNMENT_STATUS.json` no declare resuelta la alineación. No se inició otra corrida de entrenamiento en esta tanda.

## Corregido y contrastado

- Pantallas, Tailwind y Trick Room recorren protocolo → BattleState → búsqueda → tensores. Trick Room ya no borra el terreno. Tailwind modifica velocidad; Trick Room invierte el orden de velocidad respetando prioridad. Pantallas reducen el daño correspondiente, sin acumular Reflect con Aurora Veil, con excepciones para críticos e Infiltrator.
- Las pantallas del cliente usan `-1` para duración restante desconocida: el rival puede ocultar Light Clay. Se eliminan con el evento oficial; no se presenta una duración secreta como conocida. La simulación propia usa duración finita cuando conoce el estado hipotético. Clima/terreno públicos con duración desconocida también se distinguen del estado inactivo.
- Sueño: se actúa al despertar; Rest deja dos intentos dormidos; Sleep Talk consume sus propios PP, respeta las exclusiones del Dex y falla si el usuario está despierto. Parálisis y congelación tienen sorteos en las transiciones muestreadas; la búsqueda determinista conserva aproximaciones probabilísticas.
- Cambiar elimina boosts y estados temporales, aplica Regenerator/Natural Cure; un Pokémon expulsado no transmite su movimiento pendiente al reemplazo. Un pivote bloqueado por inmunidad no cambia. La elección automática de reemplazo dentro de búsqueda sigue siendo aproximada y requiere una fase explícita.
- Daño fijo respeta inmunidades de tipo. Daño realmente infligido se limita a los HP disponibles antes de calcular drenaje/recoil. Toxic escala por turnos; los orbes no dañan en el turno en que aplican el estado. No se ejecutan residuales después de terminar una partida.
- Efectos secundarios se conservan como eventos con probabilidad, no se convierten en boosts primarios. Se procesan estados, cambios de estadísticas y flinch, con Shield Dust/Covert Cloak, Serene Grace y supresión por Sheer Force. Esto no implementa todos los callbacks de habilidades o movimientos.
- Los estados, objetos, efectos de Booster Energy y habilidades revelados se identifican por partida y lado. Se evita contaminar al rival con información de un Pokémon propio de la misma especie. Cambiar o borrar boosts no borra simultáneamente los del otro lado.

## Contrato del modelo

El campo creció de 16 a 40 características (`public_field_v3`), manteniendo los primeros 16 valores. `load_compatible_state_dict` amplía con ceros las columnas del encoder de checkpoints anteriores. Los bytes originales no cambian y las características nuevas no obtienen conocimiento aprendido por esa operación. La equivalencia de salidas del encoder antiguo está probada; un entrenamiento futuro deberá aprender esas señales.

Los contratos de los experimentos anteriores no son compatibles para reanudar sobre estas observaciones: se necesita una carpeta nueva cuando se autorice reentrenar. Los resultados 73/100 y 79/100 pertenecen a la versión congelada anterior y no evalúan esta tanda.

## Evidencia

`tests/test_alignment_reference.py` ejecuta Pokémon Showdown 0.11.11, Gen 9 custom game, con sets controlados para aislar mecánicas:

- 18 secuencias deterministas comparan HP, estado, PP, boosts, Pokémon activo y duración de efectos.
- 5 familias de daño con pantallas comparan el conjunto de rolls con 256 ejecuciones oficiales por familia.
- Parálisis y congelación: 512 ejecuciones oficiales por condición, comparadas con frecuencias del motor propio.
- 7 familias de efectos secundarios: 256 ejecuciones oficiales por familia, incluyendo Nuzzle/Mortal Spin y protección de Shield Dust y Covert Cloak.

Total: 4,114 ejecuciones oficiales en esta batería. Los RNG de ambos motores son diferentes; los casos aleatorios comparan soporte/frecuencias, no identidad de la semilla. Las tolerancias se fijan en las pruebas. Estos fixtures no certifican reglas completas de OU ni todas las interacciones de Showdown.

`tests/test_public_fields.py` comprueba el recorrido a los tensores, aislamiento de información en enfrentamientos espejo y entre partidas, compatibilidad de pesos y bloqueo de entrenamiento. La suite completa pasa 171 pruebas.

## Pendiente antes de levantar la pausa

1. Completar fases explícitas de pivote/reemplazo y habilidades de entrada; la búsqueda todavía toma atajos.
2. Representar y simular Substitute, Taunt, atrapamiento, confusión y demás estados volátiles necesarios para los equipos del proyecto.
3. Validar residuales de clima/terreno, golpes múltiples y potencia dinámica. Triple Axel, Hurricane/confusión, Magma Storm/atrapamiento y Substitute ya aparecen en los equipos actuales, por lo que esto afecta al entrenamiento previsto.
4. Completar la auditoría de callbacks de objetos/habilidades, inmunidades a estados, prioridades y redondeos de modificadores combinados.
5. Contrastar trayectorias completas y estados de búsqueda con el motor oficial en todos los arquetipos actuales. La representación pública puede contener estimaciones legítimas, pero las reglas no deben generar resultados imposibles.

La paridad sigue abierta. Pasar esta batería no basta para declarar "todo corregido" ni para levantar automáticamente la pausa de entrenamiento.
