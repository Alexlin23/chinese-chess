from AlphaZero.train.benchmark import evaluate_agents, random_agent


def test_benchmark_swaps_colors_and_counts_games():
    report = evaluate_agents(random_agent, random_agent, num_games=2,
                             max_ply=4, seed=7)
    assert report.wins + report.losses + report.draws == 2
    assert sum(report.by_color['red'].values()) == 1
    assert sum(report.by_color['black'].values()) == 1
    assert 0 <= report.score_ci95[0] <= report.score_rate <= report.score_ci95[1] <= 1
