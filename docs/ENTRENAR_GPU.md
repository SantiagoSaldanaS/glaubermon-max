# Entrega para entrenamiento con NVIDIA

Código: rama `codex/simulator-training-corrections` del fork `FelipeJackFox/glaubermon-max`. No usar `main` hasta que Santiago incorpore el PR. El cierre de alineación es el de los cinco equipos de [pilot-scope.json](alignment/pilot-scope.json): dos de práctica y tres de evaluación. Se permite un piloto controlado; ampliar equipos requiere otra revisión.

La red y su optimización seleccionan CUDA automáticamente. Showdown, la preparación de estados y la búsqueda siguen consumiendo CPU; una GPU potente no garantiza acelerar toda la corrida. No hay un mínimo de VRAM medido todavía. La validación local fue en CPU (568 pruebas); la instalación CUDA siguiente debe pasar sus comprobaciones en la máquina receptora. No se ha reentrenado con las correcciones finales.

No hacen falta cuentas de Showdown, contraseñas ni el dataset de 9 GB: esta ruta genera partidas locales y parte del checkpoint que ya está en Git. No pasar `--from-scratch`. Las salidas van a una carpeta nueva bajo `runs/`; no sobrescribir `checkpoints/`.

## 1. Instalar en Linux o WSL2 (Bash)

Prerrequisitos: Git, Python 3.12 con `venv`, Node.js 22 con npm y driver NVIDIA compatible con CUDA 12.8. En WSL2 la GPU debe estar expuesta a Linux por el driver de Windows. `nvidia-smi` debe funcionar antes de continuar. Los comandos parten de una carpeta donde todavía no existe `glaubermon-max`.

```bash
git clone --branch codex/simulator-training-corrections --single-branch https://github.com/FelipeJackFox/glaubermon-max.git
cd glaubermon-max
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip install -r glaubermon/evaluation/requirements.txt
npm --prefix tools/showdown ci
python -m pip check
```

PyTorch 2.10.0/CUDA 12.8 es una distribución publicada en las [instrucciones oficiales de PyTorch](https://pytorch.org/get-started/previous-versions/). No se afirma que esta combinación se haya ejecutado en la Mac de desarrollo. Si la tarjeta/driver no soportan esta distribución, resolver esa compatibilidad antes de entrenar; no continuar silenciosamente en CPU.

## 2. Verificar máquina, pesos y autorización

Ejecutar desde la raíz del repositorio y con el entorno activado. CUDA debe imprimir `True`, el nombre de la tarjeta y completar una multiplicación en GPU. El hash comprueba el checkpoint de partida exacto. La última orden verifica versión de Showdown, equipos, esquema y código autorizados, sin entrenar.

```bash
nvidia-smi
node --version
python --version
python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.version.cuda, 'Disponible:', torch.cuda.is_available()); assert torch.cuda.is_available(), 'No hay CUDA: detenerse'; print(torch.cuda.get_device_name(0)); x=torch.randn(256,256,device='cuda'); y=x@x; torch.cuda.synchronize(); assert torch.isfinite(y).all().item(); print('Kernel CUDA OK')"
python -c "from pathlib import Path; import hashlib; p=Path('checkpoints/glaubermon_rebel_latest.pt'); h=hashlib.sha256(p.read_bytes()).hexdigest(); print(h); assert h=='2aa0deadf7d3ac650981e051544521ee8ccc0b9689f1457f921c0ed99957f0f6', 'Checkpoint inicial distinto'"
python -c "from pathlib import Path; from glaubermon.evaluation.alignment_gate import require_pilot_alignment; s=require_pilot_alignment('showdown',Path('tools/showdown/node_modules/pokemon-showdown')); print('Alineacion habilitada:', s['training_allowed'], s['clearance_source'])"
CUDA_VISIBLE_DEVICES='' python -m pytest tests/ -q -rs
```

Resultado de referencia: 568 pruebas aprobadas. No aceptar que las pruebas de Showdown se omitan por faltar el paquete. La asignación `CUDA_VISIBLE_DEVICES=''` afecta solo al comando de pruebas: reproduce su validación en CPU y no deshabilita CUDA para el entrenamiento posterior.

## 3. Piloto inicial: diez partidas

Crear una carpeta que no exista. Si existe por una ejecución anterior, pasar a la sección de continuación o elegir otro nombre y cambiarlo consistentemente en todos los comandos. No borrar el experimento anterior.

```bash
python -c "from pathlib import Path; p=Path('runs/pilot-gpu-v7-01'); p.mkdir(parents=True,exist_ok=False)"
git rev-parse HEAD > runs/pilot-gpu-v7-01/source.txt
git status --short > runs/pilot-gpu-v7-01/worktree.txt
python -m pip freeze > runs/pilot-gpu-v7-01/environment.txt
nvidia-smi > runs/pilot-gpu-v7-01/gpu.txt
set -o pipefail
python -u -m glaubermon.scripts.train_rebel --games 10 --save-every 1 --eval-every 0 --checkpoint-dir runs/pilot-gpu-v7-01 --mechanics-seed 7331 --rollout-backend showdown --showdown-path tools/showdown/node_modules/pokemon-showdown --max-turns 300 --depth 1 --torch-threads 1 2>&1 | tee runs/pilot-gpu-v7-01/train-001.log
```

El inicio debe mostrar `Initialized on: cuda` y carga de `checkpoints/glaubermon_rebel_latest.pt`. Se usa profundidad 1 para empezar; no aumentar automáticamente a 4. Después de acumular 64 muestras comienza a actualizar la red. En otra terminal se puede observar la GPU con `nvidia-smi -l 2`; utilización intermitente es esperable por el trabajo de CPU.

Comprobar metadatos, parámetros finitos y al menos un cambio real respecto del checkpoint original adaptado al esquema actual:

```bash
python -c "import json,torch; from pathlib import Path; from glaubermon.models.set_transformer import GlaubermonMaxNet; p=Path('runs/pilot-gpu-v7-01'); m=json.loads((p/'rebel_meta.json').read_text()); print(json.dumps(m,indent=2)); assert m['device'].startswith('cuda'); assert m['total_games']==10 and m['total_samples']>=64; w=torch.load(p/'glaubermon_rebel_latest.pt',map_location='cpu',weights_only=True); assert all(torch.isfinite(v).all().item() for v in w.values()); ref=GlaubermonMaxNet(); ref.load_compatible_state_dict(torch.load('checkpoints/glaubermon_rebel_latest.pt',map_location='cpu',weights_only=True)); assert any(not torch.equal(v,ref.state_dict()[k]) for k,v in w.items()); print('Pesos finitos y modificados: OK')"
```

Si falla CUDA, una prueba, una acción oficial, un timeout o aparecen NaN/Inf, detenerse y compartir el log. No desactivar `ALIGNMENT_STATUS.json` para saltar un error. Un checkpoint distinto demuestra una actualización, no una mejora de juego.

## 4. Continuar hasta cien partidas

Si el piloto anterior terminó con diez partidas y pasó sus comprobaciones, añadir noventa:

```bash
python -u -m glaubermon.scripts.train_rebel --games 90 --save-every 10 --eval-every 0 --checkpoint-dir runs/pilot-gpu-v7-01 --mechanics-seed 7331 --rollout-backend showdown --showdown-path tools/showdown/node_modules/pokemon-showdown --max-turns 300 --depth 1 --torch-threads 1 2>&1 | tee runs/pilot-gpu-v7-01/train-002.log
python -c "import json,torch; from pathlib import Path; p=Path('runs/pilot-gpu-v7-01'); m=json.loads((p/'rebel_meta.json').read_text()); print('Partidas:',m['total_games'],'Muestras:',m['total_samples'],'Dispositivo:',m['device']); assert m['total_games']==100 and m['device'].startswith('cuda'); w=torch.load(p/'glaubermon_rebel_latest.pt',map_location='cpu',weights_only=True); assert all(torch.isfinite(v).all().item() for v in w.values())"
```

**`--games` agrega partidas; no es el total objetivo**, aunque la ayuda del CLI todavía diga lo contrario. No volver a ejecutar `--games 90` suponiendo que solo completa las faltantes. Ctrl+C una vez solicita guardar y salir; esperar el mensaje de guardado. Para reanudar, conservar backend, profundidad, límite, equipos y código. Se recuperan pesos y contadores, pero el optimizador y el buffer reinician: la reanudación no es idéntica a una corrida continua.

El objetivo inicial son cien partidas, no una campaña masiva con solo dos equipos de práctica. Antes de ampliarlo, revisar fallos, tiempos y evaluación; diversificar los equipos exige ampliar la cobertura.

## 5. Evaluar el candidato congelado

Detener el entrenamiento antes de evaluar. Comparar el candidato y el checkpoint original con los mismos equipos, semillas, profundidad y límite. `--jobs 1` evita varias copias simultáneas en la GPU; `--games 100` son cien partidas **por modo**, por lo que cada comando programa 200 (hybrid + heuristic). Los equipos de evaluación no alimentan el entrenamiento nuevo.

```bash
python -m glaubermon.evaluation.official_benchmark --showdown tools/showdown/node_modules/pokemon-showdown --output runs/eval-original-v7-01 --checkpoint checkpoints/glaubermon_rebel_latest.pt --games 100 --jobs 1 --depth 2 --modes hybrid heuristic --seed 911 --max-turns 300 --decision-seconds 30
python -m glaubermon.evaluation.summarize_benchmark runs/eval-original-v7-01
python -m glaubermon.evaluation.official_benchmark --showdown tools/showdown/node_modules/pokemon-showdown --output runs/eval-candidate-v7-01 --checkpoint runs/pilot-gpu-v7-01/glaubermon_rebel_latest.pt --games 100 --jobs 1 --depth 2 --modes hybrid heuristic --seed 911 --max-turns 300 --decision-seconds 30
python -m glaubermon.evaluation.summarize_benchmark runs/eval-candidate-v7-01
```

Conservar `summary.json`, `manifest.json`, resultados individuales y trazas. Revisar partidas sin resolver, acciones inválidas, fallbacks y latencia además del winrate. El intervalo que calcula el resumidor compara hybrid contra heuristic dentro de cada corrida; no es un intervalo candidato contra original. El texto automático de `RESULTADOS.md` conserva frases del piloto histórico: usar los JSON como evidencia y redactar el informe final con las fuentes de esta corrida. Esto no equivale a ganar a Metamon/PokéChamp ni a la ladder.

## 6. Qué devolver

Compartir la carpeta `runs/pilot-gpu-v7-01/` completa (pesos, `rebel_meta.json`, logs, entorno y `rollouts/`) y las dos carpetas de evaluación. `runs/` está excluida de Git; transferir como archivo, no hacer `git add -f` de toda la carpeta.

```bash
python -c "import shutil; shutil.make_archive('entrega-pilot-gpu-v7-01','zip',root_dir='runs',base_dir='pilot-gpu-v7-01'); shutil.make_archive('entrega-eval-original-v7-01','zip',root_dir='runs',base_dir='eval-original-v7-01'); shutil.make_archive('entrega-eval-candidate-v7-01','zip',root_dir='runs',base_dir='eval-candidate-v7-01')"
```

## Windows nativo (PowerShell)

Usar Git, Python 3.12, Node.js 22/npm y driver NVIDIA instalados. Sustituir la sección 1 por:

```powershell
git -c core.autocrlf=false clone --branch codex/simulator-training-corrections --single-branch https://github.com/FelipeJackFox/glaubermon-max.git
cd glaubermon-max
git config core.autocrlf false
py -3.12 -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e '.[dev]'
python -m pip install -r glaubermon/evaluation/requirements.txt
npm --prefix tools/showdown ci
python -m pip check
```

Conservar LF (`core.autocrlf=false`) es necesario porque la autorización compara los bytes de los archivos. Los comandos de Python, Git, npm y NVIDIA de las demás secciones funcionan también en PowerShell. Para las pruebas usar:

```powershell
$env:CUDA_VISIBLE_DEVICES='-1'
python -m pytest tests/ -q -rs
Remove-Item Env:CUDA_VISIBLE_DEVICES
```

Omitir `set -o pipefail`. Usar estas líneas en lugar de los dos comandos de entrenamiento con `tee`, conservando las comprobaciones de las secciones 3/4:

```powershell
python -u -m glaubermon.scripts.train_rebel --games 10 --save-every 1 --eval-every 0 --checkpoint-dir runs/pilot-gpu-v7-01 --mechanics-seed 7331 --rollout-backend showdown --showdown-path tools/showdown/node_modules/pokemon-showdown --max-turns 300 --depth 1 --torch-threads 1 2>&1 | Tee-Object -FilePath runs/pilot-gpu-v7-01/train-001.log
if ($LASTEXITCODE -ne 0) { throw 'Fallo el piloto: revisar el log antes de continuar' }
```

Después de verificar el piloto:

```powershell
python -u -m glaubermon.scripts.train_rebel --games 90 --save-every 10 --eval-every 0 --checkpoint-dir runs/pilot-gpu-v7-01 --mechanics-seed 7331 --rollout-backend showdown --showdown-path tools/showdown/node_modules/pokemon-showdown --max-turns 300 --depth 1 --torch-threads 1 2>&1 | Tee-Object -FilePath runs/pilot-gpu-v7-01/train-002.log
if ($LASTEXITCODE -ne 0) { throw 'Fallo la continuacion: revisar el log' }
```
