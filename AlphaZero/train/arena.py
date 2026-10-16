"""Optional evaluation. Never controls learner publication or rolls weights back."""
from .benchmark import evaluate_agents, checkpoint_agent

class Arena:
    def __init__(self, config, device="cpu"):
        self.config, self.device = config, device

    def evaluate(self, candidate_path, reference_path, num_games=100, verbose=True):
        candidate = checkpoint_agent(candidate_path, self.config.num_simulations, self.device)
        reference = checkpoint_agent(reference_path, self.config.num_simulations, self.device)
        report = evaluate_agents(candidate, reference, num_games, self.config.max_game_ply,
                                 self.config.seed)
        if verbose:
            print(report)
        return report
