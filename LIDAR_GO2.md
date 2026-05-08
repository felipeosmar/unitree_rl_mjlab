# Treinamento do Go2 com LiDAR

É totalmente viável. O mjlab já tem infraestrutura de **raycasting na GPU** (Warp).
O `height_scan` que está sendo usado hoje **é tecnicamente um LiDAR**, só que num
padrão muito específico (grid paralelo apontado pra baixo). Pra um LiDAR "de verdade",
basta mudar o padrão dos rays.

## O que mjlab oferece nativamente

Em `mjlab/sensor/raycast_sensor.py`:

| Pattern                   | Geometria                              | Uso típico                            |
|---------------------------|----------------------------------------|---------------------------------------|
| `GridPatternCfg`          | Grid 2D paralelo apontando pra baixo  | Height scan (já em uso)               |
| `PinholeCameraPatternCfg` | Rays divergentes de um ponto (cônico) | Depth camera, LiDAR de estado sólido  |

Os dois compartilham o mesmo backend. Adicionar um terceiro padrão (cilíndrico 360° tipo
Velodyne, ou padrão Livox específico) é uma classe nova com ~30 linhas de código.

## Tipos de LiDAR simuláveis

### a) LiDAR de estado sólido (Livox MID-360, Avia)
Padrão *rosette* ou cônico. O Go2 EDU tem dock oficial pro Livox MID-360.

- ~200 000 pontos/s, FOV 360° horizontal × 59° vertical
- No sim: usar `PinholeCameraPatternCfg` ajustado, ou criar `LivoxPatternCfg` custom

### b) LiDAR rotativo 360° (Velodyne VLP-16, Ouster OS1)

- 16–128 canais verticais × ~1800 pontos horizontais por revolução
- No sim: criar `RotatingLidarPatternCfg(n_channels, horizontal_resolution)`
- Custo computacional alto: 32k+ rays por env por step

### c) 2D LiDAR planar (RPLiDAR, Hokuyo)

- 360 pontos num único plano horizontal
- Muito barato computacionalmente
- Limitado pra locomoção (não vê escadas / obstáculos baixos)

## Trade-offs práticos

| Aspecto                 | Height scan atual | LiDAR cônico ~5k pts | LiDAR rotativo 32k pts |
|-------------------------|-------------------|----------------------|------------------------|
| Pontos                  | 160               | ~5 000               | ~32 000                |
| VRAM por env            | desprezível       | moderada             | pesada                 |
| FPS de treino           | 100% (referência) | ~70–85%              | ~30–50%                |
| `num_envs` viável (16GB) | 4 096 (atual)    | ~2 000               | ~512–1 024             |
| Cobertura               | Só chão à frente  | Cone à frente/lados  | Esfera completa        |
| Útil pra escadas / obstáculos verticais | Limitado | Bom            | Excelente              |

## Como implementar (visão geral)

1. **Definir o padrão** — classe `LidarPatternCfg` (cilíndrica) em `mjlab/sensor/`, ou pattern
   customizado dentro do projeto.

2. **Adicionar o sensor à cena** em `src/tasks/velocity/velocity_env_cfg.py` junto com `terrain_scan`:

   ```python
   lidar_scan = RayCastSensorCfg(
     frame=ObjRef(type="body", name="base", entity="robot"),
     pattern=LidarPatternCfg(channels=16, horizontal_res=512, fov=(-15°, 15°)),
     max_distance=20.0,
   )
   ```

3. **Adicionar como observação** no `actor_terms` (com ruído pra sim-to-real).

4. **Arquitetura da política**:
   - **MLP simples** funciona pra LiDAR pequeno (até ~1024 pontos), mas é ineficiente.
   - **PointNet** ou **CNN 1D** sobre a nuvem ordenada (canal × azimute) — padrão na literatura.
   - **Voxelização + sparse 3D conv** pra LiDAR denso.

5. **Modelar ruído realista**:
   - Distância: ±0.02–0.05 m típico
   - Drop-out de pontos (alguns rays não retornam)
   - Falha em superfícies absorventes/reflexivas (vidro, espelho)

## Quando vale a pena vs quando não

**Vale**:
- Navegação em ambientes 3D (escadas, móveis, paredes)
- Desvio de obstáculos verticais
- Localização (SLAM)
- Você já tem o sensor físico no robô (Go2 EDU + Livox)

**Não vale**:
- Locomoção em terreno plano ou rough sem obstáculos verticais — o `height_scan` atual é
  mais que suficiente e muito mais barato.
- Quer iterar rápido — LiDAR adiciona horas no tempo de treino.
- VRAM apertada (a RTX 2000 com 16 GB já está em ~13 GB com 4096 envs e height_scan).

## Hardware: o LiDAR já está no Go2 de fábrica

Todas as versões do Go2 (base/AIR, PRO, EDU) vêm com o **Unitree 4D LiDAR L1**
montado na cabeça, com varredura frontal:

| Spec                     | Valor                                  |
|--------------------------|----------------------------------------|
| Tecnologia               | Solid-state, 4D (3D + intensidade)     |
| Range                    | ~30 m                                  |
| FOV                      | 360° horizontal × 59° vertical (spec)  |
| Scan rate                | 5–15 Hz                                |
| Pontos/s                 | ~21 600                                |
| Pontos por step (a 20 Hz)| ~1 080                                 |

Versões com sensores adicionais:

- **Go2 EDU**: mantém o L1 + dock pra Livox MID-360 (mais denso) e outros add-ons
- **Go2 PRO / EDU**: também tem câmera Intel RealSense (depth)

## Estado atual no treino deste projeto

**O L1 NÃO está no observation stack atual.** O único sensor raycast configurado em
`src/tasks/velocity/velocity_env_cfg.py` é o `terrain_scan` (grid downward 16×10 = 160
pontos). Faz sentido pro treino de locomoção, mas significa que a política não enxerga
nada à frente.

Pra incluir o L1, seria preciso adicionar em `velocity_env_cfg.py` algo do tipo:

```python
lidar_l1 = RayCastSensorCfg(
  frame=ObjRef(type="body", name="head", entity="robot"),  # ou imu body
  pattern=LidarPatternCfg(
    horizontal_fov=(0, 360),
    vertical_fov=(-29.5, 29.5),  # 59° total
    channels=32,                  # ajustar pra ~1080 pts
    horizontal_res=64,
  ),
  max_distance=30.0,
)
```

Adicionar como `actor_term` com ruído `Unoise(±0.05)` simulando o erro do sensor real.

## Resumindo

A infraestrutura já está pronta. A pergunta é: **vale a pena pro caso de uso?**

- Pra locomoção em terreno (treino atual): `height_scan` é melhor (mais barato, suficiente).
- Pra navegação ou ambientes 3D complexos: LiDAR é essencial.
