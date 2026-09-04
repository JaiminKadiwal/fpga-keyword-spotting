# Models

Large checkpoints are ignored by Git. Keep reusable notes and lightweight export
metadata here.

Suggested local folders:

```text
models/checkpoints/   # ignored PyTorch checkpoints
models/exported/      # lightweight fixed-point exports, .mem files, golden vectors
```

The first target model is an int8 MLP:

```text
490 -> 128 -> 64 -> 12
```
