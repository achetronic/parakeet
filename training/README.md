# Exporta los assets ONNX de Parakeet desde el checkpoint oficial de NVIDIA

Genera todo lo que el servidor necesita para no depender de conversiones de
terceros: encoder (atención completa y local), decoder+joint, vocab, config y
el preprocessor mel.

## Requisitos

- Python 3.12 (NO 3.14: torch/nemo no lo soportan).
- ~20 GB de disco (checkpoint 2.5 GB + assets fp32/int8).
- `uv` para crear el venv.

## Poner en marcha el venv (una vez)

```sh
cd training
uv venv --python 3.12 venv
source venv/bin/activate
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu126
uv pip install -r requirements.txt
```

## Generar los assets

```sh
source training/venv/bin/activate
python training/export.py --checkpoint training/checkpoint/parakeet-tdt-0.6b-v3.nemo --out training/assets
```

Si el checkpoint no existe, se descarga solo desde
`nvidia/parakeet-tdt-0.6b-v3` (no requiere token).

Produce en `training/assets/`:

| archivo                         | qué es                                              |
| ------------------------------- | --------------------------------------------------- |
| `config.json`                   | config del modelo                                    |
| `vocab.txt`                     | SentencePiece vocab + `<blk>` (8193 líneas)          |
| `nemo128.onnx`                  | preprocessor mel (referencia de tests)               |
| `encoder-model.onnx` + `.data`  | encoder fp32, atención COMPLETA                      |
| `encoder-model.int8.onnx`       | encoder int8, atención COMPLETA                      |
| `encoder-local.onnx` + `.data`  | encoder fp32, atención LOCAL (`rel_pos_local_attn`)  |
| `encoder-local.int8.onnx`       | encoder int8, atención LOCAL                         |
| `decoder_joint-model.onnx`      | decoder+joint fp32                                   |
| `decoder_joint-model.int8.onnx` | decoder+joint int8                                   |
| `manifest.txt`                  | sha256 de cada archivo                               |

## Atención local: el parche que hay que conocer

NeMo 3.0 no exporta bien la atención local (`RelPositionMultiHeadAttentionLongformer`):
su ruta de sliding-chunks reconstruye la banda con `new_empty()` + asignaciones
por `slice`, que el exportador torchscript compila mal y dan salida basura.

`export.py` resuelve esto reemplazando el forward por una atención completa +
máscara en banda, equivalente numéricamente (ma x diff ~1e-3) y con solo
operaciones exportables.

Caveat: esa implementación es O(T²) en memoria, no O(T·w) como el
sliding-chunks. Vale para audios de decenas de minutos; para horas habría que
arreglar el export O(n) del sliding-chunks.

## nemo128.onnx (pendiente)

`nemo128.onnx` es solo una referencia de tests (el servidor NO lo carga; la
validación del frontend mel la hace `internal/mel`). El script lo exporta en
modo best-effort: con torch >= 2.x el `torch.stft` devuelve complejos y el
exportador torchscript no los traza, así que de momento se omite. Para
regenerarlo habría que parchear el `stft` del preprocessor a `return_complex=False`.
Mientras tanto se reusa el `nemo128.onnx` del repo de referencia solo como
soporte de tests.

## Publicar los assets (HuggingFace)

Los assets no se versionan en git (pesan >2 GB). Se suben a un repo HF propio y
el Makefile los descarga desde ahí:

```sh
huggingface-cli upload <tu-user>/parakeet-tdt-0.6b-v3-onnx training/assets/ . --repo-type model
```

Luego apunta el Makefile a ese repo:

```sh
make MODELS_REPO=<tu-user>/parakeet-tdt-0.6b-v3-onnx models-int8
```