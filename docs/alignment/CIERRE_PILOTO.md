# Criterio de cierre para el primer piloto

Alcance acordado: corregir los equipos actuales y pasar a un piloto controlado. El objetivo de este cierre no es implementar todo Pokémon Showdown. La lista exacta y los hashes de equipos están en [pilot-scope.json](pilot-scope.json): 57 movimientos, 16 habilidades y 13 objetos de `development_offense`, `development_rain`, `balance`, `stall` y `pelol`. Cambiar esos sets requiere revisar la cobertura.

## 1. Historial y observación

Resuelto y probado: Struggle como menú temporal separado del moveset; PP propios conservados por partida/Pokémon, gasto observado, Pressure y movimientos llamados; duraciones públicas inferibles de Encore/Disable/Taunt; activación y origen de Protosynthesis/Quark Drive. Checkpoints anteriores se adaptan en memoria y no se sobrescriben.

Cuando falta historial, los PP y la fuente de una activación se marcan como desconocidos. No se inventan PP exactos ni temporizadores ocultos. En un inicio de observación con solo Struggle y sin causa conocida, la búsqueda conserva ese menú de forma conservadora hasta recibir datos. HP de Substitute y otros datos privados siguen siendo hipótesis; esto debe distinguirse de una regla implementada incorrectamente.

## 2. Mecánicas del inventario

Tienen contraste directo con Showdown: Protosynthesis/Quark Drive, Body Press, Protean, Roost, Thunderclap, Water Absorb, Flash Fire y Lum Berry; teracristalización y reentrada de Ogerpon-Wellspring; cadenas de modificadores de Wellspring Mask, Black Glasses, Supreme Overlord, Vessel of Ruin, Multiscale y Life Orb; combinaciones de Choice Band/Specs y Paradox; orden de cambios simultáneos con Drizzle, Booster Energy y Dauntless Shield.

Pendiente: cerrar la revisión explícita de todas las entradas del inventario, vinculando cada una con evidencia de regresión o pruebas diferenciales. La aparición de un nombre en el código no basta para certificar la regla ni todas sus interacciones.

## 3. Comprobación conjunta y salida

Comparar trayectorias oficiales de todos los arquetipos fijados con las decisiones legales y transiciones de la búsqueda. Separar diferencias por información oculta/aleatoriedad de errores deterministas de reglas. Las discrepancias deterministas relevantes deben quedar corregidas y convertidas en regresiones. Completar partidas por el adaptador real a profundidad 2 sin acciones inválidas, fallbacks ni timeouts; registrar código, pesos, semillas y trazas.

Solo después de cerrar estos puntos se revisa `training_allowed`. El primer entrenamiento crea otro experimento/checkpoint; no sustituye los respaldos. La calidad del checkpoint nuevo se evalúa después con el control congelado. Superar Metamon/PokéChamp y la evaluación de ladder son hitos posteriores.
