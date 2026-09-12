# Alineación con Showdown: abierta; reentrenamiento pausado

Instrucción de Felipe/Santiago: corregir las diferencias de entorno antes de reentrenar. `AlphaZeroTrainer.train()` rechaza corridas mientras `docs/ALIGNMENT_STATUS.json` no permita entrenar. No se inició reentrenamiento en estas tandas; las pruebas de optimizador usan modelos desechables. Los checkpoints originales se conservan.

## Corregido en la tercera tanda

- **Clima y terreno:** Rain Dance/Sunny Day/Sandstorm/Snowscape y los cuatro terrenos, setters de entrada (incluido Drizzle), duración de rocas/Terrain Extender, expiración, Cloud Nine/Air Lock, residuales y recuperación dependiente del clima. Grassy Terrain recupera HP al terminar el turno; Psychic Terrain bloquea prioridad dirigida a objetivos en tierra; Misty/Electric conservan sus restricciones de estado/Rest. Ice Spinner elimina también el contador del terreno.
- **Velocidad:** Swift Swim, Chlorophyll, Sand Rush, Slush Rush, Surge Surfer y Quick Feet se combinan con etapas, objetos, Tailwind, parálisis y Trick Room. Prankster/Gale Wings aportan prioridad; se contempla la inmunidad Dark a estado con Prankster.
- **Daño y precisión:** bonificaciones de terreno, reducción de Earthquake en Grassy y Dragon en Misty, defensas de Rock con arena/Ice con nieve, Weather Ball/Terrain Pulse y precisión de Hurricane/Thunder/Blizzard según el clima. La prueba diferencia daño directo, críticos y residuales.
- **Golpes múltiples:** carga de metadatos canónicos, cantidad de impactos y precisión por golpe, Skill Link/Loaded Dice, potencia 20/40/60 de Triple Axel y 10/20/30 de Triple Kick. Cada impacto actualiza HP, Substitute, contacto y efectos secundarios; el ataque se detiene por fallo o KO y Life Orb cobra una vez. En búsqueda determinista se usa una cantidad representativa de golpes; las transiciones muestreadas sí sortean la distribución.

Estas correcciones no certifican todos los callbacks ni redondeos combinados. Sigue pendiente representar como creencias los temporizadores ocultos y las activaciones ambientales persistentes de Protosynthesis/Quark Drive. El contrato de tensores continúa en `public_volatile_v4`.

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

`public_volatile_v4` conserva los 40 campos globales y amplía los datos por Pokémon de 64 a 68 valores para Substitute, Taunt, ligadura y confusión. Los campos globales 37–39 identifican quién debe reemplazar y si existe una continuación interna del turno. `load_compatible_state_dict` amplía los pesos anteriores con columnas cero y pone a cero las nuevas señales de fase al importar el formato anterior. Los bytes originales no se modifican; esta operación no enseña al modelo las nuevas reglas.

Un futuro entrenamiento necesita un experimento nuevo; no se reanuda una carpeta con un contrato de observación anterior. Los resultados 73/100 y 79/100 pertenecen a la versión congelada `01a6ee1` y no evalúan estas correcciones.

## Evidencia automatizada

Referencia: paquete fijado **Pokémon Showdown 0.11.11**, Gen 9 custom game, con sets/estados controlados para aislar mecánicas. Los fixtures toman instantáneas independientes; no reutilizan objetos mutables del motor oficial.

- Primera batería: 18 secuencias deterministas, 5 familias de daño con pantallas (256 batallas cada una), parálisis/congelación (512 cada una) y 7 familias de efectos secundarios (256 cada una).
- Segunda batería: 45 secuencias deterministas de volátiles/reemplazos/entrada, 2 familias de confusión (512 cada una) y 2 de Magma Storm (256 cada una).
- Tercera batería: 77 secuencias deterministas, 12 familias de daño (256 cada una), 6 de cantidad de impactos (512 cada una), Triple Axel por impacto (256) y 5 de precisión por clima (512 cada una).
- Total: **14,732 ejecuciones oficiales**. Se comparan HP, estado, PP, boosts, objetos, volátiles, hazards, jugador que debe reemplazar y contador de turno. En los casos aleatorios se comparan soportes/frecuencias con tolerancias explícitas, no resultados idénticos por semilla: los RNG son diferentes.
- Suite completa: **328 pruebas aprobadas**, incluyendo protocolo→tensores, aislamiento entre lados, compatibilidad de pesos, independencia de ramas y bloqueo de entrenamiento.

Además, se completaron 16 partidas locales de integración sin acciones inválidas, fallbacks ni timeouts. Cuatro corresponden a la tercera tanda: dos entre equipos de desarrollo y dos de control a profundidad 2. Triple Axel no fue elegido en esas partidas; se valida por impacto en fixtures. En la comparación histórica de la tanda anterior, la evaluación por lotes conservó las decisiones de las dos partidas comparadas y redujo el máximo observado de 26.787 a 15.534 s. Detalle y fuentes: [INTEGRACION.md](INTEGRACION.md).

Estos fixtures controlados no certifican reglas completas de OU ni equivalencia general entre motores.

## Pendiente antes de levantar la pausa

1. Ampliar cobertura de fases a callbacks de entrada simultáneos y otros pivotes (Baton Pass, Shed Tail, Parting Shot, Teleport). Las tres familias de pivote usadas por los equipos de desarrollo ya tienen fases explícitas.
2. Completar otros volátiles relevantes (Encore, Disable, Leech Seed). En observación pública, HP restante de Substitute y temporizadores ocultos son desconocidos: la búsqueda aún usa hipótesis conservadoras, no una distribución de creencias completa. La fuerza de Binding Band oculta también se estima. Una solicitud pública de reemplazo no revela el movimiento enemigo pendiente; el cliente evalúa esa situación sin copiar la cola privada del motor oficial.
3. Ampliar clima/terreno a setters simultáneos, climas primigenios, activaciones persistentes de Protosynthesis/Quark Drive y otros callbacks de potencia dinámica. Drizzle/Swift Swim y Triple Axel ya están implementados y contrastados en los casos controlados descritos.
4. Completar callbacks de objetos/habilidades, inmunidades, prioridades y redondeos combinados. La existencia de un helper de entrada no implica que todas las habilidades estén implementadas.
5. Contrastar trayectorias completas y estados de búsqueda en todos los arquetipos de desarrollo, y evaluar calidad de juego de esta versión.

La paridad sigue abierta. Ni esta batería ni una prueba de integración levantan automáticamente la pausa del entrenamiento.
