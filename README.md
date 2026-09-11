> Estado de las correcciones y comandos del piloto oficial: [docs/TRAINING_CORRECTIONS.md](docs/TRAINING_CORRECTIONS.md). El entrenamiento actual usa observaciones del cliente y resultados de Showdown; el motor interno sigue siendo experimental. Los checkpoints originales se conservan.

# Glaubermon Max

Glaubermon Max is an expert/superhuman game-theoretic AI for **Pokémon Showdown**.

## Key Innovations
1. **Bayesian Particle Filter & Belief State Engine**: Accurately tracks hidden moves, abilities, items, EV spreads, and Tera types.
2. **Exact Reverse Damage Inversion**: Deduces stat benchmarks and boosts from damage rolls.
3. **Simultaneous-Move Nash Matrix Solver**: Solves normal-form matrix games via Linear Programming to output unexploitable mixed strategies ($\pi^*$).
4. **Permutation-Invariant Set Transformer**: Multi-head self- and cross-attention representation for 6v6 team structures.
5. **AlphaStar-Style League Training**: Multi-agent population-based training with Prioritized Fictitious Self-Play.
