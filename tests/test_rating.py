from owars.utils import Rating, update_ratings


def test_winner_mu_increases_loser_decreases():
    a = Rating(mu=600.0, sigma=200.0)
    b = Rating(mu=600.0, sigma=200.0)
    a2, b2 = update_ratings(a, b, score_a=1.0)
    assert a2.mu > a.mu
    assert b2.mu < b.mu
    assert a2.sigma <= a.sigma
    assert b2.sigma <= b.sigma


def test_draw_pulls_toward_mean():
    a = Rating(mu=700.0, sigma=100.0)
    b = Rating(mu=500.0, sigma=100.0)
    a2, b2 = update_ratings(a, b, score_a=0.5)
    assert a2.mu < a.mu
    assert b2.mu > b.mu


def test_sigma_floor():
    a = Rating(mu=600.0, sigma=200.0)
    b = Rating(mu=600.0, sigma=200.0)
    for _ in range(50):
        a, b = update_ratings(a, b, score_a=1.0)
    assert a.sigma >= 30.0
    assert b.sigma >= 30.0
