# Alineación con Showdown: abierta; reentrenamiento pausado

Instrucción de Felipe/Santiago: corregir las diferencias de entorno antes de reentrenar. `AlphaZeroTrainer.train()` rechaza corridas mientras `docs/ALIGNMENT_STATUS.json` no permita entrenar. No se inició reentrenamiento en estas tandas; las pruebas de optimizador usan modelos desechables. Los checkpoints originales se conservan.

## Cierre acotado del piloto

El cierre se limita a los equipos actuales: [criterios y pendientes concretos](CIERRE_PILOTO.md). El inventario fijado incluye 57 movimientos, 16 habilidades y 13 objetos. Las mecánicas ajenas a esos sets ya no bloquean por sí mismas este primer piloto.

## Séptima tanda: modificadores, Ogerpon y entradas simultáneas

- Wellspring Mask y Black Glasses modifican potencia base; la máscara aumenta también Horn Leech y Play Rough. Supreme Overlord usa el modificador fijo de potencia correspondiente a 0–5 aliados caídos, junto con Black Glasses.
- Choice Band/Specs, Paradox, Flash Fire y Vessel of Ruin combinan modificadores antes de redondear la estadística. Pantallas, Multiscale y Life Orb comparten la cadena de modificación final, incluido el valor fijo 5324/4096 de Life Orb.
- Ogerpon-Wellspring pierde Water Absorb al adquirir Embody Aspect, aumenta Defensa Especial una vez y vuelve a recibir ese aumento al reingresar. El cliente cuenta solo el evento público de aumento; el nombre público abreviado de la habilidad conserva su variante correcta.
- Dos cambios voluntarios usan la velocidad de los Pokémon salientes (incluido Trick Room) para ordenar callbacks. Se contrastan Drizzle/Damp Rock, entrada con Paradox/Booster Energy y Dauntless Shield en ambos lados.
- Evidencia añadida: 22 familias de daño con 256 batallas cada una y 10 secuencias de Tera/entrada; **25,847 ejecuciones oficiales controladas** acumuladas. Suite completa: **487 pruebas aprobadas**. Continúa `public_callbacks_v7` y la pausa de entrenamiento. Falta la revisión conjunta de cobertura/trayectorias de los cinco equipos.

## Sexta tanda: callbacks de los equipos actuales

- Thunderclap/Sucker Punch verifican que el rival tenga un ataque pendiente; fallan frente a estado, cambio y después de actuar. Water Absorb cura antes de precisión/Substitute y detiene Flip Turn. Flash Fire absorbe también estado, conserva el aumento ofensivo y se limpia al cambiar.
- Lum Berry se consume y cura estado/confusión antes de la siguiente acción, entre impactos y después de Toxic Spikes; no se consume a través de Substitute ni después de un KO.
- Protean cambia el tipo antes de Protect/precisión, una vez por entrada; un tipo ya coincidente no gasta la activación. Roost elimina Flying durante ese turno, conserva la fase interrumpida por pivote y respeta Tera. Los tipos originales se conservan separados de los actuales; ambos influyen correctamente en STAB.
- Contrato `public_callbacks_v7`: 86 características por Pokémon, con tres señales nuevas (Protean usado, Roost y Flash Fire). Protocolo → estado → tensores conserva las señales y el aislamiento; importar pesos de 83 columnas agrega ceros sin reescribir archivos. Estas señales todavía no están aprendidas.
- Evidencia: 40 secuencias oficiales y cuatro familias de daño de 256 batallas (Flash Fire físico/especial y Protean antes/después de usarlo). Suite completa: **454 pruebas aprobadas**. Total: **20,205 ejecuciones oficiales controladas**, más la secuencia del adaptador para PP/Struggle. Las tandas sexta y séptima se integraron juntas en las cinco partidas de `984a42c`; ver INTEGRACION.md.

## Quinta tanda: historial, activaciones y Body Press

- Struggle deja de reemplazar el moveset propio. Se conserva el historial de PP por partida/Pokémon, incluidos gasto observado, Pressure y exclusión de movimientos llamados. El menú temporal sigue siendo autoritativo para la acción actual. Una observación tardía sin PP previos conserva incertidumbre; no inventa agotamiento ni disponibilidad futura.
- Encore, Disable y Taunt llevan duración finita cuando el registro público permite inferirla, incluyendo antes/después de actuar y reemplazos. Una secuencia oficial completa reproduce Encore + Disable → Struggle → recuperación de Surf conservando los PP originales.
- Protosynthesis/Quark Drive fijan la estadística al activarse, distinguen campo de Booster Energy, consumen el objeto cuando corresponde y conservan/eliminan el efecto al cambiar el campo o el Pokémon. Cloud Nine y expiración del campo tienen casos controlados. Se corrigió el redondeo del aumento del 30%.
- Body Press toma Defensa y sus etapas, pero aplica modificadores ofensivos de Ataque. Se contrastan Iron Defense/boosts, Choice Band, Eviolite, Fur Coat, Unaware y críticos; no se trata como un ataque basado en Ataque normal.
- Contrato `public_history_v6`: 33 características por movimiento y 83 por Pokémon. El nuevo campo indica si los PP son conocidos o estimados. Los encoders de movimientos históricos de 32 columnas reciben una columna cero en memoria; no se modifica el checkpoint ni se atribuye aprendizaje a esa adaptación.

## Corregido en la cuarta tanda

- **Encore y Disable:** conservan movimiento afectado y duración, restringen acciones sin destruir PP y expiran. Encore sustituye una selección ya encolada conservando su prioridad; Disable la cancela antes de gastar PP. Se contemplan ausencia de movimiento previo, PP agotados, flags canónicos de exclusión de Encore, cambio, Substitute, Magic Bounce, Aroma Veil y Mental Herb en casos controlados. Encore junto con Disable puede obligar a Struggle.
- **Leech Seed:** inmunidad Grass, bloqueo por Substitute, reflejo, Magic Guard, Liquid Ooze, Big Root y limpieza con Rapid Spin/cambio. La curación sigue el puesto activo del lado que sembró y no la identidad del Pokémon que salió. Se aplica después de recuperación de objetos/Grassy Terrain y antes de daño de estado; se valida orden entre dos objetivos sembrados con y sin Trick Room. Al terminar la partida no se siguen descontando temporizadores.
- **Observación pública:** último movimiento revelado por Pokémon, Encore/Disable con movimiento afectado y temporizador desconocido, y lado que recibe Leech Seed. Se conserva aislamiento de partidas, lados y ramas de búsqueda. Una restricción del request deja de sobrescribir PP con cero; el filtro temporal se separa del agotamiento real.
- **Modelo:** contrato `public_restrictions_v5`, 83 características por Pokémon. Añade presencia de las tres condiciones y referencias al movimiento encorado, deshabilitado y último usado. Importa pesos de 64/68 características con columnas nuevas en cero; los pesos originales no se reescriben. Estas columnas aún no están aprendidas.

## Corregido en la tercera tanda

- **Clima y terreno:** Rain Dance/Sunny Day/Sandstorm/Snowscape y los cuatro terrenos, setters de entrada (incluido Drizzle), duración de rocas/Terrain Extender, expiración, Cloud Nine/Air Lock, residuales y recuperación dependiente del clima. Grassy Terrain recupera HP al terminar el turno; Psychic Terrain bloquea prioridad dirigida a objetivos en tierra; Misty/Electric conservan sus restricciones de estado/Rest. Ice Spinner elimina también el contador del terreno.
- **Velocidad:** Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Surge Surfer y Quick Feet se combinan con etapas, objetos, Tailwind, parálisis y Trick Room. Prankster/Gale Wings aportan prioridad; se contempla la inmunidad Dark a estado con Prankster.
- **Daño y precisión:** bonificaciones de terreno, reducción de Earthquake en Grassy y Dragon en Misty, defensas de Rock con arena/Ice con nieve, Weather Ball/Terrain Pulse y precisión de Hurricane/Thunder/Blizzard según el clima. La prueba diferencia daño directo, críticos y residuales.
- **Golpes múltiples:** carga de metadatos canónicos, cantidad de impactos y precisión por golpe, Skill Link/Loaded Dice, potencia 20/40/60 de Triple Axel y 10/20/30 de Triple Kick. Cada impacto actualiza HP, Substitute, contacto y efectos secundarios; el ataque se detiene por fallo o KO y Life Orb cobra una vez. En búsqueda determinista se usa una cantidad representativa de golpes; las transiciones muestreadas sí sortean la distribución.

Estas correcciones no certifican todos los callbacks ni redondeos combinados. Sigue pendiente representar como creencias los temporizadores ocultos y las activaciones ambientales persistentes de Protosynthesis/Quark Drive. Esta tercera tanda conservó el contrato `public_volatile_v4`; la cuarta lo amplía a `public_restrictions_v5`.

## Corregido en la segunda tanda

- **Substitute:** coste de HP, fallo por HP insuficiente o sustituto existente, absorción sin transferir daño sobrante, bloqueo de estados/efectos dirigidos, excepciones de sonido/Infiltrator/bypasssub y eliminación al cambiar. Conserva callbacks que sí atraviesan el sustituto: Air Balloon, limpieza y aumento de velocidad de Rapid Spin; Knock Off no retira el objeto protegido. Drenaje redondea hacia arriba.
- **Taunt:** filtra acciones de búsqueda y detiene movimientos de estado ya seleccionados antes de consumir PP; duración distinta cuando llega antes/después de actuar, con Magic Bounce, Mental Herb y varias inmunidades. El sueño/congelación se procesan antes, de acuerdo con la prioridad del motor oficial.
- **Atrapamiento:** Magma Storm y movimientos de ligadura conservan fuente, duración, residuales y efectos de Grip Claw/Binding Band. La restricción desaparece al salir la fuente, expirar, crear Substitute o limpiar con giro. Se representan Mean Look/Block/Spider Web; Ghost/Shed Shell conservan sus excepciones para cambiar. Los pivotes pueden salir de una ligadura.
- **Confusión:** duración por intentos de actuar, expiración previa a la acción, 33% de autogolpe en transiciones muestreadas, fórmula propia sin aplicar Choice Band ni STAB y sin gastar PP al golpearse. Se transporta desde eventos públicos y efectos secundarios como Hurricane.
- **Fases de reemplazo:** U-turn/Volt Switch/Flip Turn interrumpen el turno y piden una elección real. Se conserva la cola de movimientos, su usuario original, PP, Protect y flinch. El ataque pendiente alcanza al Pokémon entrante; el que ya actuó no vuelve a atacar. KOs simultáneos y KOs por hazards requieren sus respectivas decisiones, sin repetir residuales ni avanzar el turno prematuramente. El cambio de perspectiva también conserva fuentes de ligadura y la cola pendiente. La búsqueda evalúa todas las opciones de reemplazo y agrupa sus hojas en un lote para la red; ya no manda automáticamente el primer Pokémon vivo de la banca.
- **Entrada:** mínimo de un punto de daño de hazards, Toxic Spikes (incluida absorción por Poison), Sticky Web y Heavy-Duty Boots; Intimidate en casos controlados y Dauntless Shield/Intrepid Sword una sola vez por batalla. Esto no certifica todos los callbacks de entrada.
- **Cliente y modelo:** registra volátiles por partida/lado/especie, los elimina al cambiar y conserva las activaciones únicas de habilidades de entrada. Los estados desconocidos siguen marcados como desconocidos. Las solicitudes oficiales de reemplazo llegan a la búsqueda y a las características del modelo.

La regresión antigua que exigía Close Combat sobre Rapid Spin en un final de Great Tusk era incompatible con aplicar hazards al reemplazo. La prueba ahora reproduce ambos caminos: quitar hazards permite sobrevivir al último Ogerpon de 16 HP; conservarlas causa su KO al entrar. No se añadió una bonificación artificial para imponer la decisión anterior.

## Correcciones de la primera tanda (`68c218b`)

Pantallas, Tailwind y Trick Room recorren protocolo → estado → búsqueda → tensores. Pantallas respetan categoría, críticos e Infiltrator; Trick Room no borra terreno. Se corrigieron sueño/Rest/Sleep Talk, probabilidades de parálisis/congelación, inmunidades a daño fijo, Toxic, orbes, daño realmente infligido, efectos secundarios, limpieza de boosts al cambiar y aislamiento de objetos/habilidades/Booster Energy revelados entre lados y partidas.

## Contrato del modelo

`public_history_v6` añade el indicador de PP conocidos al encoder de movimientos (32 → 33); conserva 40 campos globales y amplía los datos por Pokémon a 83 valores: 64 históricos, 4 de Substitute/Taunt/ligadura/confusión, 3 presencias de Encore/Disable/Leech Seed y 12 indicadores del movimiento afectado/último movimiento (cuatro posiciones para cada uno). Los campos globales 37–39 identifican quién debe reemplazar y si existe una continuación interna del turno. `load_compatible_state_dict` importa los formatos de 64 y 68 características mediante columnas cero; solo al importar 64 se ponen a cero también las señales de fase. Se aceptan tensores históricos con padding explícito. No se modifican los bytes originales ni se enseña al modelo las nuevas reglas con esta adaptación.

Un futuro entrenamiento necesita un experimento nuevo; no se reanuda una carpeta con un contrato de observación anterior. Los resultados 73/100 y 79/100 pertenecen a la versión congelada `01a6ee1` y no evalúan estas correcciones.

## Evidencia automatizada

Referencia: paquete fijado **Pokémon Showdown 0.11.11**, Gen 9 custom game, con sets/estados controlados para aislar mecánicas. Los fixtures toman instantáneas independientes; no reutilizan objetos mutables del motor oficial.

- Primera batería: 18 secuencias deterministas, 5 familias de daño con pantallas (256 batallas cada una), parálisis/congelación (512 cada una) y 7 familias de efectos secundarios (256 cada una).
- Segunda batería: 45 secuencias deterministas de volátiles/reemplazos/entrada, 2 familias de confusión (512 cada una) y 2 de Magma Storm (256 cada una).
- Tercera batería: 77 secuencias deterministas, 12 familias de daño (256 cada una), 6 de cantidad de impactos (512 cada una), Triple Axel por impacto (256) y 5 de precisión por clima (512 cada una).
- Cuarta batería: 40 secuencias deterministas de restricciones y Leech Seed, más 512 batallas para precisión de Leech Seed. Incluye PP, último movimiento, expiración, casos de KO, prioridad y efecto de Trick Room sobre residuales simultáneos.
- Quinta batería: 17 secuencias de activación Paradox, 8 familias de daño Paradox (256 cada una), 5 familias de Body Press (256 cada una) y una de críticos (512). Adicionalmente se prueba una secuencia oficial por el adaptador real para historial/Struggle.
- Total hasta la quinta tanda: **19,141 ejecuciones oficiales**. Se comparan HP, estado, PP, boosts, objetos, volátiles, hazards, jugador que debe reemplazar y contador de turno. En los casos aleatorios se comparan soportes/frecuencias con tolerancias explícitas, no resultados idénticos por semilla: los RNG son diferentes.
- Suite de la quinta tanda: **407 pruebas aprobadas**, incluyendo protocolo→tensores, aislamiento entre lados, compatibilidad de pesos, independencia de ramas y bloqueo de entrenamiento.

Además, se completaron 29 partidas locales de integración sin acciones inválidas, fallbacks ni timeouts. Cinco corresponden a `984a42c`, incluyen los cinco equipos fijados y 341 solicitudes con menús/PP activos coincidentes. Cuatro corresponden a la quinta tanda, con movimientos de 33 características y pesos congelados (30/27 turnos de desarrollo y 21/24 de control). Cuatro corresponden a la cuarta tanda, con observaciones de 83 características y pesos congelados. Sus equipos no llevan Encore/Disable/Leech Seed: la validación mecánica proviene de fixtures. Otras cuatro corresponden a la tercera tanda: dos entre equipos de desarrollo y dos de control a profundidad 2. Triple Axel no fue elegido en esas partidas; se valida por impacto en fixtures. En la comparación histórica de la tanda anterior, la evaluación por lotes conservó las decisiones de las dos partidas comparadas y redujo el máximo observado de 26.787 a 15.534 s. Detalle y fuentes: [INTEGRACION.md](INTEGRACION.md).

Estos fixtures controlados no certifican reglas completas de OU ni equivalencia general entre motores.

## Pendiente antes de levantar la pausa

Los pendientes que bloquean el primer piloto están enumerados en [CIERRE_PILOTO.md](CIERRE_PILOTO.md) y `docs/ALIGNMENT_STATUS.json`: completar la auditoría del inventario actual, entradas simultáneas y comparación conjunta de trayectorias. Reglas ajenas al inventario quedan diferidas.

La paridad general sigue abierta. Ninguna batería ni prueba de integración levanta automáticamente la pausa del entrenamiento.
