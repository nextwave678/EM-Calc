"""
Quick validation for SkewProbabilityEngine.
Tests symmetric case, known bearish skew, and known bullish skew.
"""
import sys
sys.path.insert(0, '.')

from gvc_engine import SkewProbabilityEngine

def test_symmetric():
    """When put_iv == call_iv, everything should be symmetric."""
    spe = SkewProbabilityEngine()
    result = spe.full_asymmetric_profile(em_base=5.0, put_iv=0.20, call_iv=0.20)

    assert result['p_up'] == 0.50, f"p_up should be 0.50, got {result['p_up']}"
    assert result['p_down'] == 0.50, f"p_down should be 0.50, got {result['p_down']}"
    assert result['bias_direction'] == 'neutral', f"bias should be neutral, got {result['bias_direction']}"
    # em_up and em_down should both equal em_base when symmetric
    assert abs(result['em_up'] - 5.0) < 0.01, f"em_up should be 5.0, got {result['em_up']}"
    assert abs(result['em_down'] - 5.0) < 0.01, f"em_down should be 5.0, got {result['em_down']}"
    # No GEX adjustment in neutral
    assert abs(result['em_up_final'] - 5.0) < 0.01
    assert abs(result['em_down_final'] - 5.0) < 0.01
    print("✅ Symmetric case PASSED")


def test_bearish_skew():
    """put_iv=0.25, call_iv=0.15 → P(down)=62.5%, bearish bias, EM_down > EM_up."""
    spe = SkewProbabilityEngine()
    result = spe.full_asymmetric_profile(em_base=5.0, put_iv=0.25, call_iv=0.15)

    assert abs(result['p_down'] - 0.625) < 0.01, f"p_down should be ~0.625, got {result['p_down']}"
    assert abs(result['p_up'] - 0.375) < 0.01, f"p_up should be ~0.375, got {result['p_up']}"
    assert result['bias_direction'] == 'bearish', f"bias should be bearish, got {result['bias_direction']}"
    assert result['em_down'] > result['em_up'], "em_down should be > em_up for bearish skew"
    assert result['downside_gex'] == 'negative', f"downside_gex should be negative, got {result['downside_gex']}"
    assert result['upside_gex'] == 'positive', f"upside_gex should be positive, got {result['upside_gex']}"
    # GEX adjustment: downside expanded (1.1x), upside compressed (0.9x)
    assert result['em_down_final'] > result['em_down'], "em_down_final should be expanded"
    assert result['em_up_final'] < result['em_up'], "em_up_final should be compressed"
    print(f"✅ Bearish skew PASSED — P(down)={result['p_down']:.1%}, EM↓=${result['em_down_final']:.2f}, EM↑=${result['em_up_final']:.2f}")


def test_bullish_skew():
    """call_iv > put_iv → bullish bias."""
    spe = SkewProbabilityEngine()
    result = spe.full_asymmetric_profile(em_base=5.0, put_iv=0.15, call_iv=0.25)

    assert result['bias_direction'] == 'bullish', f"bias should be bullish, got {result['bias_direction']}"
    assert result['p_up'] > result['p_down'], "p_up should be > p_down"
    assert result['em_up'] > result['em_down'], "em_up should be > em_down"
    assert result['upside_gex'] == 'negative', f"upside_gex should be negative, got {result['upside_gex']}"
    assert result['downside_gex'] == 'positive', f"downside_gex should be positive, got {result['downside_gex']}"
    print(f"✅ Bullish skew PASSED — P(up)={result['p_up']:.1%}, EM↑=${result['em_up_final']:.2f}, EM↓=${result['em_down_final']:.2f}")


if __name__ == '__main__':
    test_symmetric()
    test_bearish_skew()
    test_bullish_skew()
    print("\n🎉 All tests passed!")
