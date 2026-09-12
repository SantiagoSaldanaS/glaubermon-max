# Cierre del primer piloto: correcciones verificadas

Alcance acordado: corregir los equipos actuales y pasar a un piloto controlado. El objetivo de este cierre no es implementar todo Pokémon Showdown. La lista exacta y los hashes de equipos están en [pilot-scope.json](pilot-scope.json): 57 movimientos, 16 habilidades y 13 objetos de `development_offense`, `development_rain`, `balance`, `stall` y `pelol`. Cambiar esos sets requiere revisar la cobertura.

## 1. Historial y observación

Resuelto y probado: Struggle como menú temporal separado del moveset; PP propios conservados por partida/Pokémon, gasto observado, Pressure y movimientos llamados; duraciones públicas inferibles de Encore/Disable/Taunt; activación y origen de Protosynthesis/Quark Drive. Checkpoints anteriores se adaptan en memoria y no se sobrescriben.

Cuando falta historial, los PP y la fuente de una activación se marcan como desconocidos. No se inventan PP exactos ni temporizadores ocultos. En un inicio de observación con solo Struggle y sin causa conocida, la búsqueda conserva ese menú de forma conservadora hasta recibir datos. HP de Substitute y otros datos privados siguen siendo hipótesis; esto debe distinguirse de una regla implementada incorrectamente.

## 2. Mecánicas del inventario

Tienen contraste directo con Showdown: Protosynthesis/Quark Drive, Body Press, Protean, Roost, Thunderclap, Water Absorb, Flash Fire y Lum Berry; teracristalización y reentrada de Ogerpon-Wellspring; cadenas de modificadores de Wellspring Mask, Black Glasses, Supreme Overlord, Vessel of Ruin, Multiscale y Life Orb; combinaciones de Choice Band/Specs y Paradox; orden de cambios simultáneos con Drizzle, Booster Energy y Dauntless Shield.

Revisión explícita cerrada en [pilot-coverage.json](pilot-coverage.json): 57 comparaciones de movimientos con los sets reales y evidencia específica para las 16 habilidades y 13 objetos. No se certifican todas las interacciones posibles fuera de estos equipos.

## 3. Comprobación conjunta y salida

Cinco trayectorias completas con resultados aleatorios fijados coinciden de principio a fin, incluidos los menús legales; 292 fases y 248 turnos. Las discrepancias encontradas quedaron corregidas y cubiertas por regresiones. Las pruebas de distribuciones siguen separadas, y la incertidumbre de observaciones públicas no se sustituye por información privada. Integración final de `cab3367` completada: dos partidas por el adaptador real a profundidad 2 y tres a profundidad 1, sin acciones inválidas, fallbacks ni timeouts. La reproducción de 341 solicitudes coincide en menús legales y PP activos. Código, pesos, semillas y hashes están en [pilot-closure-results.json](pilot-closure-results.json) y [integration-results.json](integration-results.json).

Con 568 pruebas aprobadas y la integración final verificada, `training_allowed` queda habilitado para el piloto acotado por el motor oficial. La entrada al entrenador verifica versión, esquema, equipos y hashes de implementación. No se inició reentrenamiento ni se actualizaron pesos; los ocho originales permanecen intactos. El primer entrenamiento crea otro experimento/checkpoint; no sustituye los respaldos. La calidad del checkpoint nuevo se evalúa después con el control congelado. Superar Metamon/PokéChamp y la evaluación de ladder son hitos posteriores.
