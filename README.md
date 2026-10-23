# Pipeline de dataset YOLO (vídeo → CVAT → YOLO → Colab)

Classes (ordem = id YOLO): `amortecedor_vibracao`, `esfera_sinalizacao`, `isolador_bastao`, `isolador_disco`.
Tudo é configurado em `configs/pipeline.toml`.

## 1. Instalação (Windows)

Abra a pasta no VS Code e, no terminal (PowerShell):

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Se der erro de política de execução ao ativar: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.

> **Python 3.14:** o código só precisa de `opencv-python`, `numpy` e `matplotlib` (sem PyTorch localmente).
> Se algum pacote ainda não tiver wheel para 3.14 e a instalação falhar, instale o Python 3.12
> ao lado (`py -3.12 -m venv .venv`). O treino roda no Colab, que tem seu próprio Python.

Extensões recomendadas (o VS Code sugere ao abrir): Python, Jupyter e **Colab** (Google).

## 2. Fluxo

| Passo | Comando (ou Terminal → Run Task) | O que faz |
|---|---|---|
| 1 | `python pipeline.py extract` | lê `data/videos/*`, filtra e seleciona frames → `data/frames/selected/` + `manifest.csv` |
| 2 | `python pipeline.py pack` | gera `data/cvat_upload/lote_XX.zip` (`batch_size` imagens cada), `cvat_labels.json` e `LEIA-ME_CVAT.txt` |
| — | **CVAT (navegador)** | anotar e exportar cada lote como **YOLO 1.1** (sem imagens) para `data/cvat_export/` |
| 3 | `python pipeline.py build` | valida labels, divide **por vídeo** (70/20/10), grava `data/dataset/` e gera `reports/dataset_report.html` |
| 4 | `python pipeline.py sync` | cria `dist/dataset.zip` e copia para a pasta do Drive |
| — | `notebooks/train_colab.ipynb` | treino com kernel Colab (GPU) |
| 5 | `python pipeline.py fetch` | traz `best.pt`, curvas e métricas do Drive para `runs/` |

### Etapa 1: seleção de frames
Cascata: amostragem (4 fps) → descarte de escuros → descarte de borrados (nitidez < 50% do percentil 75 do vídeo)
→ deduplicação em sequência (dHash; mantém o frame mais nítido de cada trecho parado) → seleção por
diversidade entre todos os vídeos (farthest-point sampling). O resumo por vídeo aparece no terminal e em
`reports/extracao.csv`; o motivo de cada descarte está em `manifest.csv`.

Para ajustar a quantidade: `hash_threshold` (maior = menos frames), `feature_min_distance` (maior = menos frames),
`max_frames` (teto por execução). Depois, `python pipeline.py extract --force`.
**Faça isso antes de começar a anotar.** `--force` refaz tudo e invalida o que já foi anotado.

### Adicionando vídeos depois de anotar
Coloque os vídeos novos em `data/videos/` e rode, **sem** `--force`:

```powershell
python pipeline.py extract                  # só lê os vídeos que não estão no manifest
python pipeline.py pack --batch-size 600    # só empacota frames sem lote: lote_03, lote_04, ...
```

Frames, status e números de lote já existentes não mudam, então os lotes já anotados no CVAT continuam
valendo. A seleção dos frames novos evita imagens parecidas com as já selecionadas. `--max-frames N`
limita quantos frames novos entram nesta execução (padrão: `max_frames`).
Dica: `keep_rejected = true` guarda os descartados em `data/frames/rejected/<motivo>/` para você auditar visualmente.

### Etapa 2: CVAT
1. Crie um Project e cole `cvat_labels.json` na aba **Raw** dos labels.
2. Uma tarefa por zip (`My computer`).
3. Imagem sem objeto: deixe sem anotar (vira fundo). Meta: ≥ 5%.
4. Exportar: *Export task dataset → YOLO 1.1*, **sem** "Save images".
5. Se o limite de armazenamento do plano gratuito apertar, use `max_side = 1280` em `[cvat]`: só os zips ficam menores;
   os labels são normalizados e continuam valendo para as imagens originais.

### Etapa 3: o que o `build` garante
- Classes remapeadas **por nome** (a ordem do export não precisa coincidir).
- Erros (classe inexistente, linha malformada) interrompem; avisos (caixa fora da imagem, duplicada) são corrigidos e listados em `reports/problemas_anotacao.csv`.
- Em um lote exportado, frame sem `.txt` ou com `.txt` vazio = fundo. Lotes ainda não exportados ficam **fora** do dataset.
- Split por vídeo, escolhido por busca aleatória (seed fixa) para aproximar 70/20/10 em imagens **e** em instâncias por classe.
- Com poucos vídeos o equilíbrio fica imperfeito; o relatório avisa. Mude `[split].seed` para outra combinação.
- Para imagens de fundo, grave um trecho só com o cenário.

### Etapa 4: Colab no VS Code
1. Ajuste `[colab].drive_dir` para a pasta do Google Drive para computador (ex.: `G:/Meu Drive/yolo_tcc`) e rode `sync`.
2. Abra `notebooks/train_colab.ipynb` → seletor de kernel → **Colab → New Colab Server** → GPU (T4).
3. Rode as células em ordem. O notebook copia o dataset do Drive para o disco local do Colab, treina o YOLO11n
   (salvando em `Drive/.../runs/<run_name>`), e avalia o **teste** uma vez no final com o critério mAP@0.5 ≥ 0,80.
4. Se o runtime cair, rode tudo de novo: o treino retoma de `last.pt`.
5. Localmente: `python pipeline.py fetch`.

O runtime do Colab não enxerga arquivos locais; por isso a ponte é o Google Drive.

## 3. Limitações conhecidas
- Testado de ponta a ponta com vídeos sintéticos e um export de CVAT simulado. **Não** foi testado com o CVAT real,
  com o Python 3.14 nem com o Colab real; confira as primeiras execuções.
- `drive.mount` pode se comportar diferente na extensão do VS Code do que no navegador. Se falhar, avise para
  trocarmos para outra forma de transferência.
- Limiares de blur/dedup/diversidade foram calibrados com dados sintéticos; ajuste com seus vídeos olhando o resumo da extração.
