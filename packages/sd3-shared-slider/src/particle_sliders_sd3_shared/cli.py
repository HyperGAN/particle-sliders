"""Use the shared lifecycle CLI; no product update loop or checkpoint envelope."""
def main(argv=None):
    from particle_sliders.session_cli import main as shared_main
    return shared_main(argv)
