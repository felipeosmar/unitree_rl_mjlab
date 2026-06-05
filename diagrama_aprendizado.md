# Diagrama de Blocos — Aprendizado por Reforço do Unitree Go2 com LiDAR

```mermaid
flowchart TB
    subgraph Ambiente["🌍 Ambiente (MuJoCo + Warp GPU)"]
        direction TB
        Sim["Simulador Físico<br/>MuJoCo mj_step()<br/>dt = 5ms × 4 = 20ms"]
        Terr["Terreno<br/>Plano (Flat)"]
        DR["Randomização<br/>- Massa<br/>- Atrito<br/>- CoM<br/>- Push externo<br/>- Ruído de encoder"]
    end

    subgraph Robo["🤖 Robô — Unitree Go2"]
        direction TB
        Base["Base Link<br/>(corpo principal)"]
        Joints["Articulações (12)<br/>- 4 por perna<br/>- Abdução/Quedrilho/Junho<br/>(posição + velocidade)"]
        Pés["Pés (4)<br/>- Contato solo<br/>- Força de contato<br/>- Altura do pé<br/>- Air time"]
    end

    subgraph Sensores["📡 Sensores"]
        direction TB
        IMU["IMU (base)<br/>- Velocidade angular (3)<br/>- Gravidade projetada (3)<br/>- Velocidade linear (3)"]
        LIDAR["LiDAR L1<br/>- 32 canais × 64 res = 2048 raios<br/>- 360° horizontal<br/>- 59° vertical<br/>- Alcance 30m"]
        Encoders["Encoders<br/>- Posição juntas (12)<br/>- Velocidade juntas (12)"]
        Contacts["Contato<br/>- Sensor de contato (4)<br/>- Força nos pés (4×3)"]
    end

    subgraph Observações["👁️ Observações (state space)"]
        direction TB
        ObsActor["Actor Observation Space (d=2094)<br/>- base_ang_vel (3)<br/>- projected_gravity (3)<br/>- command — vel alvo (3)<br/>- phase — fase da marcha (1)<br/>- joint_pos (12)<br/>- joint_vel (12)<br/>- last_action (12)<br/>- lidar_scan (2048)<br/><br/>Ruído aditivo U[-0.05, 0.05]"]
        ObsCritic["Critic Observation Space (d=2121)<br/>Tudo do actor +<br/>- base_lin_vel (3)<br/>- foot_height (4)<br/>- foot_air_time (4)<br/>- foot_contact (4)<br/>- foot_contact_forces (12)<br/><br/>**Sem ruído** (critic vê estado real)"]
        Noise["Corrupção (só actor)<br/>Ruído uniforme por termo<br/>Simula imperfeições reais"]
    end

    subgraph Comando["🎮 Comando de Velocidade"]
        Cmd["Command Manager<br/>- vx alvo (m/s)<br/>- vy alvo (m/s)<br/>- ωz alvo (rad/s)<br/><br/>Amostrado a cada 4s<br/>Distribuição uniforme"]
    end

    subgraph Marcha["🦶 Fase da Marcha (Gait Phase)"]
        Phase["Phase Oscillator<br/>Período = 0.6s (trote)<br/>Onda senoidal<br/>Fase = t / período<br/><br/>Sincronização das pernas<br/>Trote: diagonais alternadas"]
    end

    subgraph Policy["🧠 Política (Actor-Critic)"]
        direction TB
        Actor["Actor (MLP)<br/>1024 → 512 → 256 → 12<br/>Ativação: ELU<br/>Saída: média da Gaussiana"]
        Critic["Critic (MLP)<br/>1024 → 512 → 256 → 1<br/>Ativação: ELU<br/>Saída: valor V(s)"]
        Dist["Distribuição Gaussiana<br/>- std treinável escalar<br/>- Amostragem estocástica<br/>  (treino)<br/>- Média determinística<br/>  (inferência)"]
        ObsNorm["Normalização<br/>Running Mean + Std<br/>Por observação"]
    end

    subgraph Ação["🎯 Ação (Action Space)"]
        Action["12 posições alvo de juntas<br/>(target joint positions)<br/>- FR: 3 juntas<br/>- FL: 3 juntas<br/>- RR: 3 juntas<br/>- RL: 3 juntas<br/><br/>Clipadas em [-1, 1]<br/>Convertidas para ângulos<br/>via PD controller"]
        PD["PD Controller<br/>- Ganhos fixos<br/>- τ = kp(q_des − q) + kd(0 − q̇)<br/>- Torques aplicados no MuJoCo"]
    end

    subgraph Recompensa["🏆 Recompensa (Reward Function)"]
        direction TB
        R1["track_lin_vel: seguir vx/vy alvo"]
        R2["track_ang_vel: seguir ωz alvo"]
        R3["body_orientation: manter base nivelada"]
        R4["joint_limits: evitar limites das juntas"]
        R5["action_rate: suavidade da ação"]
        R6["foot_gait: sincronia do trote"]
        R7["foot_slip: evitar escorregar"]
        R8["soft_landing: reduzir impacto"]
        Penalty["Penalidades:<br/>- Queda (fell_over)<br/>- Contato ilegal<br/>- Momento angular<br/>→ ENCERRAM EPISÓDIO"]
    end

    subgraph Algoritmo["🧪 Algoritmo PPO"]
        direction TB
        Buffer["Rollout Buffer<br/>48 steps × 16384 envs<br/>= 786k transições<br/>GPU (cuda)"]
        GAE["GAE (λ=0.95, γ=0.99)<br/>Vantagem = erro de valor<br/>suavizado no tempo"]
        Epochs["8 épocas de treino<br/>16 mini-batches cada<br/>= 128 updates por iteração"]
        Loss["Loss = L_policy + c₁L_value − c₂H_entropy<br/>- L_policy: PPO clipped surrogate<br/>- L_value: MSE do valor<br/>- H_entropy: entropia da política"]
        Grad["Gradient Clipping (max=1.0)<br/>Adam optimizer<br/>LR = 1e-3 (adaptive)"]
    end

    subgraph Iteracao["🔄 Uma Iteração de Treino"]
        direction LR
        Coleta["Coleta (collect)<br/>48 passos por ambiente<br/>16384 ambientes em GPU"]
        Aprendizado["Aprendizado (learn)<br/>128 mini-batches<br/>~8 épocas"]
    end

    subgraph Logger["📊 Logging"]
        WB["Weights & Biases<br/>- Reward components<br/>- Episode length<br/>- FPS / tempo<br/>- Checkpoints a cada 100 iterações"]
    end

    %% Conexões principais
    Sim --> |"estado físico"| Robo
    Robo --> |"lê sensores"| Sensores
    Sensores --> |"raw data"| Observações
    Noise --> |"corrompe"| ObsActor
    Cmd --> |"vx, vy, ωz"| ObsActor
    Phase --> |"fase 0-1"| ObsActor
    Observações --> |"obs tensor"| ObsNorm
    ObsNorm --> |"normalizado"| Policy

    Actor --> |"média + std"| Dist
    Dist --> |"amostragem"| Ação
    Action --> |"q_des (12)"| PD
    PD --> |"τ (torques)"| Sim

    Critic --> |"V(s)"| GAE

    %% Loop de coleta
    Ação -.-> |"próximo step"| Sim
    Sim --> |"novo estado"| Sensores

    %% Coleta no buffer
    ObsActor --> Buffer
    Dist --> |"log_prob"| Buffer
    Action --> Buffer
    Critic --> |"V(s)"| Buffer
    Recompensa --> |"rewards"| Buffer
    Buffer --> GAE
    GAE --> |"advantages + returns"| Epochs
    Epochs --> Loss
    Loss --> Grad
    Grad --> |"atualiza pesos"| Actor
    Grad --> |"atualiza pesos"| Critic

    %% Domain randomization
    DR --> |"perturba"| Sim

    %% Timing da iteração
    Iteracao --> |"19.7s total"| WB

    %% Terminal conditions
    Recompensa --> |"done = caiu ou timeout"| Sim
    Sim --> |"reset se done"| Robo

    %% Estilo
    classDef robot fill:#4a90d9,color:#fff,stroke:#2a5a8a
    classDef sensor fill:#7b68ee,color:#fff,stroke:#5a48be
    classDef obs fill:#2ecc71,color:#fff,stroke:#1a8a4a
    classDef policy fill:#e67e22,color:#fff,stroke:#b85e0a
    classDef action fill:#e74c3c,color:#fff,stroke:#b53a2a
    classDef reward fill:#f39c12,color:#fff,stroke:#c37a0a
    classDef algo fill:#9b59b6,color:#fff,stroke:#7a3a96
    classDef sim fill:#1abc9c,color:#fff,stroke:#128a6a
    classDef log fill:#34495e,color:#fff,stroke:#1a2a3a
    class Robo,Base,Joints,Pés robot
    class Sensores,IMU,LIDAR,Encoders,Contacts sensor
    class Observações,ObsActor,ObsCritic,Noise obs
    class Policy,Actor,Critic,Dist,ObsNorm policy
    class Ação,Action,PD action
    class Recompensa,R1,R2,R3,R4,R5,R6,R7,R8,Penalty reward
    class Algoritmo,Buffer,GAE,Epochs,Loss,Grad algo
    class Ambiente,Sim,Terr,DR sim
    class Logger,WB log
```

## Legenda dos Conceitos

| Conceito | Descrição |
|---|---|
| **State Space** | Conjunto de observações que o robô percebe: IMU, juntas, LiDAR, comando, fase |
| **Action Space** | 12 posições alvo de juntas (3 por perna × 4 pernas) |
| **Policy (Actor)** | Rede neural que mapeia observação → ação. MLP com 3 camadas ocultas |
| **Critic** | Rede neural que estima o valor V(s) — "quão bom é este estado" |
| **PPO** | Proximal Policy Optimization — algoritmo de RL que limpa o tamanho do update por iteração |
| **GAE** | Generalized Advantage Estimation — calcula a vantagem de cada ação com suavização temporal |
| **Rollout** | Coleta de N passos de interação antes de cada update do policy |
| **Decimation** | 4 passos de simulação MuJoCo para 1 passo de política (50 Hz → 200 Hz física) |
| **Domain Randomization** | Variação aleatória de parâmetros da simulação para transferir para o mundo real (sim-to-real) |
| **Gait Phase** | Oscilador que sincroniza as pernas no padrão de trote (diagonais alternadas) |
| **Reward Shaping** | Função de recompensa composta por múltiplos termos que guiam o comportamento desejado |

## Pipeline de Dados (visão simplificada)

```
Simulação → Sensores → Observações → Actor → Ação → PD Controller → Torques → Simulação
                                        ↓
                     Critic → V(s) → Vantagem (GAE) → PPO Update → novos pesos
```
