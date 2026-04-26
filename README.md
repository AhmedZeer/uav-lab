![UAV Lab Thumbnail](assets/uav-ab-thumb.png)

# UAV Lab
End-to-end software stack to train fixed-wing UAV policies.
Check the [demo!](https://youtu.be/vJGbIgS7SR4?si=85nzMvPeHLRmRvV5)

![UAV Lab Demo](assets/demo1.gif)

## Software Stack
```mermaid
flowchart TB
    subgraph L1[Operator and Learning Interfaces]
        direction LR
        UI[User Interfaces<br/>Manual setpoint UI<br/>Follow camera UI<br/>Debug scripts]
        TP[Training and Playback<br/>skrl train/play<br/>RL checkpoints]
    end

    subgraph L2[Decision Layer]
        direction LR
        ENV[Task Environments<br/>Dogfight / multi-UAV scenarios<br/>Observations, rewards, resets]
        POL[RL Policies and Commands<br/>Policy outputs<br/>Manual throttle, roll, pitch setpoints]
    end

    subgraph L3[Aircraft Simulation]
        direction LR
        CTRL[Flight Controller<br/>Fixed-wing autopilot]
        AERO[Aerodynamics Engine<br/>Lift, drag, moments, prop thrust]
        FM[Flight Mechanics<br/>Rigid-body state and force application]
    end

    subgraph L4[Scene and Runtime]
        direction LR
        SCENE[Scene and Assets<br/>Terrain, lighting, UAV USD assets]
        ISAAC[Isaac Lab / Isaac Sim<br/>Scene orchestration and rendering]
        PHYSX[PhysX Backend<br/>Rigid-body simulation]
    end

    UI --> ENV
    TP --> ENV
    ENV --> POL
    POL --> CTRL
    CTRL --> AERO
    AERO --> FM
    FM --> SCENE
    SCENE --> ISAAC
    ISAAC --> PHYSX

    style L1 fill:#e8f4ff,stroke:#3b82f6,stroke-width:1px
    style L2 fill:#ecfdf3,stroke:#16a34a,stroke-width:1px
    style L3 fill:#fff3e0,stroke:#f97316,stroke-width:1px
    style L4 fill:#f3e8ff,stroke:#8b5cf6,stroke-width:1px
```

## Stack Layers

- `User Interfaces`: Manual control panels, follow-camera controls, and visualization scripts.
- `Training and Playback`: skrl-based training, policy checkpointing, and policy replay.
- `Task Environments`: Multi-UAV task definitions, observations, rewards, resets, and termination logic.
- `Guidance and Control`: RL/manual setpoints and fixed-wing autopilot loops.
- `Aircraft Simulation`: Flight mechanics, aerodynamics, and force/torque application.
- `Scene and Assets`: Isaac Lab scene configuration, terrain, lighting, and UAV assets.
- `Simulation Runtime`: Isaac Sim, Isaac Lab, and PhysX execution backend.
