import pytest


def test_collect_dispatch_help(capsys):
    pytest.importorskip('particle_sliders_vibe')
    from particle_sliders.session_cli import main
    with pytest.raises(SystemExit) as exc:
        main(['collect', '--help'])
    assert exc.value.code == 0
    assert '--vibe' in capsys.readouterr().out
